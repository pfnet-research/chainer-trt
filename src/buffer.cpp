/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <algorithm>
#include <cassert>
#include <sstream>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/profiling.hpp"

#include "include/chainer_trt_impl.hpp"

chainer_trt::buffer::buffer(std::shared_ptr<model> _m, int _batch_size)
  : m(_m), batch_size(_batch_size) {
    nvtx_profile_color("buffer::ctor", -3) {
        if(m->get_max_batch_size() < batch_size)
            throw std::range_error("batch_size");

        std::vector<std::vector<int>> dims;
        for(int i = 0; i < m->get_n_inputs(); ++i)
            dims.push_back(m->get_input_dimensions(i));
        for(int i = 0; i < m->get_n_outputs(); ++i)
            dims.push_back(m->get_output_dimensions(i));

        for(const auto dim : dims) {
            const int n_vals = chainer_trt::internal::calc_n_elements(dim);
            const int n_bytes = n_vals * sizeof(float) * batch_size;
            void* p;
            cudaMalloc(&p, n_bytes);
            gpu_buffers.push_back(p);
            gpu_buffer_sizes.push_back(n_bytes);
        }
    }
}

chainer_trt::buffer::buffer(const buffer&& that)
  : m(that.m), batch_size(that.batch_size), gpu_buffers(that.gpu_buffers),
    gpu_buffer_sizes(that.gpu_buffer_sizes) {
    gpu_buffers.clear();
    gpu_buffer_sizes.clear();
}

chainer_trt::buffer::~buffer() {
    nvtx_profile_color("buffer::dtor", -3) {
        for(void* p : gpu_buffers)
            cudaFree(p);
    }
}

int chainer_trt::buffer::get_batch_size() const {
    return batch_size;
}

void chainer_trt::buffer::input_host_to_device(
  const std::vector<const void*>& inputs_cpu, cudaStream_t stream) {
    if((int)inputs_cpu.size() != m->get_n_inputs()) {
        std::stringstream oss;
        oss << "The number of inputs doesn't match (";
        oss << "model has " << m->get_n_inputs() << " inputs,";
        oss << "specified " << inputs_cpu.size() << std::endl;
        throw std::invalid_argument(oss.str());
    }
    nvtx_profile_color("input_host_to_device", -4) {
        for(size_t i = 0; i < inputs_cpu.size(); ++i)
            cudaMemcpyAsync(gpu_buffers[i], inputs_cpu[i], gpu_buffer_sizes[i],
                            cudaMemcpyHostToDevice, stream);
    }
}

void chainer_trt::buffer::input_host_to_device(
  const std::map<std::string, const void*>& inputs_cpu, cudaStream_t stream) {
    int correct_input_cnt = 0;
    for(auto it : inputs_cpu) {
        if(!m->has_input(it.first))
            continue;
        int bind_idx = m->engine->getBindingIndex(it.first.c_str());
        cudaMemcpyAsync(gpu_buffers[bind_idx], it.second,
                        gpu_buffer_sizes[bind_idx], cudaMemcpyHostToDevice,
                        stream);
        correct_input_cnt++;
    }
    if(correct_input_cnt != m->get_n_inputs()) {
        std::ostringstream oss;
        oss << "The network has the following inputs:" << std::endl;
        for(auto name : m->get_input_names())
            oss << "- " << name << std::endl;
        oss << "Whereas only the following inputs are specified:" << std::endl;
        for(auto it : inputs_cpu)
            oss << "- " << it.first << std::endl;
        throw std::invalid_argument(oss.str());
    }
}

void chainer_trt::buffer::output_device_to_host(
  const std::vector<void*>& outputs_cpu, cudaStream_t stream) const {
    if((int)outputs_cpu.size() != m->get_n_outputs())
        internal::throw_n_in_out_err(outputs_cpu.size(), m->get_n_outputs(),
                                     "outputs");

    nvtx_profile_color("output_device_to_host_async", -4) {
        for(size_t i = 0, gpu_buffer_idx = m->get_n_inputs();
            i < outputs_cpu.size(); ++i, ++gpu_buffer_idx)
            cudaMemcpyAsync(outputs_cpu[i], gpu_buffers[gpu_buffer_idx],
                            gpu_buffer_sizes[gpu_buffer_idx],
                            cudaMemcpyDeviceToHost, stream);
    }
}

void chainer_trt::buffer::output_device_to_host(
  const std::map<std::string, void*>& outputs_cpu, cudaStream_t stream) const {
    int correct_output_cnt = 0;
    for(auto it : outputs_cpu) {
        if(!m->has_output(it.first))
            continue;
        int bind_idx = m->engine->getBindingIndex(it.first.c_str());
        cudaMemcpyAsync(it.second, gpu_buffers[bind_idx],
                        gpu_buffer_sizes[bind_idx], cudaMemcpyDeviceToHost,
                        stream);
        correct_output_cnt++;
    }
    if(correct_output_cnt != m->get_n_outputs()) {
        std::ostringstream oss;
        oss << "The network has the following outputs:" << std::endl;
        for(auto name : m->get_output_names())
            oss << "- " << name << std::endl;
        oss << "Whereas only the following outputs are specified:" << std::endl;
        for(auto it : outputs_cpu)
            oss << "- " << it.first << std::endl;
        throw std::invalid_argument(oss.str());
    }
}

int chainer_trt::buffer::get_buffer_size(int buffer_idx,
                                         bool with_batch) const {
    return gpu_buffer_sizes[buffer_idx] / (with_batch ? 1 : batch_size);
}

int chainer_trt::buffer::get_input_size(int input_idx, bool with_batch) const {
    if(input_idx < 0 || m->get_n_inputs() <= input_idx)
        throw std::invalid_argument("The specified input index out of range");
    return get_buffer_size(input_idx, with_batch);
}

int chainer_trt::buffer::get_input_size(const std::string& input_name,
                                        bool with_batch) const {
    const int input_idx = m->engine->getBindingIndex(input_name.c_str());
    if(input_idx < 0 || m->get_n_inputs() <= input_idx)
        throw std::invalid_argument("Input " + input_name + " not found");
    return get_input_size(input_idx, with_batch);
}

int chainer_trt::buffer::get_output_size(int output_idx,
                                         bool with_batch) const {
    // buffer: [x0, x1, x2, y0, y1, y2]   (x*: user inputs, y*: outputs)
    // output index in buffer is (output idx) + (# user inputs)
    if(output_idx < 0 || m->get_n_outputs() <= output_idx)
        throw std::invalid_argument("The specified output index out of range");
    return get_buffer_size(output_idx + m->get_n_inputs(), with_batch);
}

int chainer_trt::buffer::get_output_size(const std::string& output_name,
                                         bool with_batch) const {
    // buffer: [x0, x1, x2, y0, y1, y2]
    // output index is (binding idx) - (# inputs)
    const int bind_idx = m->engine->getBindingIndex(output_name.c_str());
    const int output_idx = bind_idx - m->get_n_inputs();
    if(output_idx < 0 || m->get_n_outputs() <= output_idx)
        throw std::invalid_argument("Output " + output_name + " not found");
    return get_output_size(output_idx, with_batch);
}

void* chainer_trt::buffer::get_input(int input_idx) const {
    if(input_idx < 0 || m->get_n_inputs() <= input_idx)
        throw std::invalid_argument("The specified input index out of range");
    return gpu_buffers[input_idx];
}

void* chainer_trt::buffer::get_input(const std::string& input_name) const {
    // input index is same as binding index
    const int input_idx = m->engine->getBindingIndex(input_name.c_str());
    if(input_idx < 0 || m->get_n_inputs() <= input_idx)
        throw std::invalid_argument("Input " + input_name + " not found");
    return gpu_buffers[input_idx];
}

void* chainer_trt::buffer::get_output(int output_idx) const {
    if(output_idx < 0 || m->get_n_outputs() <= output_idx)
        throw std::invalid_argument("The specified output index out of range");
    return gpu_buffers[m->get_n_inputs() + output_idx];
}

void* chainer_trt::buffer::get_output(const std::string& output_name) const {
    const int bind_idx = m->engine->getBindingIndex(output_name.c_str());
    const int output_idx = bind_idx - m->get_n_inputs();
    if(output_idx < 0 || m->get_n_outputs() <= output_idx)
        throw std::invalid_argument("Output " + output_name + " not found");
    return gpu_buffers[output_idx + m->get_n_inputs()];
}
