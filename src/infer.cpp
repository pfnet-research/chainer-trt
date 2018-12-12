/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/profiling.hpp"
#include "include/chainer_trt_impl.hpp"

namespace chainer_trt {

infer::infer(std::shared_ptr<model> _m,
             std::shared_ptr<nvinfer1::IProfiler> _profiler)
  : m(_m), profiler(_profiler) {
    auto cxt = m->engine->createExecutionContext();
    cuda_context =
      internal::unique_ptr_with_destroyer<nvinfer1::IExecutionContext>(cxt);
    if(profiler)
        cuda_context->setProfiler(profiler.get());
}

std::shared_ptr<buffer> infer::create_buffer(int batch_size) const {
    return std::make_shared<buffer>(m, batch_size);
}

std::vector<void*> infer::create_bindings(
  const std::map<std::string, void*> name_buffer_map) const {
    std::vector<void*> bindings(m->engine->getNbBindings(), nullptr);

    // fill inputs and outputs
    for(auto p = name_buffer_map.begin(); p != name_buffer_map.end(); ++p) {
        const std::string& name = p->first;
        void* ptr = p->second;

        const int bind_idx = m->engine->getBindingIndex(name.c_str());
        if(bind_idx != -1)
            bindings[bind_idx] = ptr;
        else if(debug)
            std::cerr << "[debug] ignored " << name
                      << ", because the NN doesn't have an output with "
                      << "the specified name" << std::endl;
    }

    // check if nothing is left behind
    for(unsigned i = 0; i < bindings.size(); ++i) {
        if(!bindings[i]) {
            auto msg = std::string(m->engine->getBindingName(i)) +
                       " is not specified in name_buffer_map";
            throw std::invalid_argument(msg);
        }
    }
    return bindings;
}

std::vector<void*>
infer::create_bindings(const std::vector<void*>& inputs_gpu,
                       const std::vector<void*>& outputs_gpu) const {
    std::vector<void*> bindings;
    if((int)inputs_gpu.size() != m->get_n_inputs())
        internal::throw_n_in_out_err(inputs_gpu.size(), m->get_n_inputs(),
                                     "inputs");
    if((int)outputs_gpu.size() != m->get_n_outputs())
        internal::throw_n_in_out_err(outputs_gpu.size(), m->get_n_outputs(),
                                     "outputs");

    for(int i = 0; i < m->get_n_inputs(); ++i)
        bindings.push_back(inputs_gpu[i]);
    for(int i = 0; i < m->get_n_outputs(); ++i)
        bindings.push_back(outputs_gpu[i]);
    return bindings;
}

int infer::get_binding_index(const std::string& name) const {
    if(!m->has_input(name) && !m->has_output(name))
        throw std::invalid_argument(
          name + " not found in neither NN input nor output");
    return m->engine->getBindingIndex(name.c_str());
}

void infer::operator()(buffer& buf, cudaStream_t stream) {
    nvtx_profile_color("infer::operator()", -1) {
        cuda_context->enqueue(buf.batch_size, buf.gpu_buffers.data(), stream,
                              NULL);
    }
}

void infer::infer_from_cpu(int batch_size,
                           const std::vector<const void*>& inputs_cpu,
                           const std::vector<void*>& outputs_cpu,
                           cudaStream_t stream) {
    if((int)inputs_cpu.size() != m->get_n_inputs())
        internal::throw_n_in_out_err(inputs_cpu.size(), m->get_n_inputs(),
                                     "inputs");
    if((int)outputs_cpu.size() != m->get_n_outputs())
        internal::throw_n_in_out_err(outputs_cpu.size(), m->get_n_outputs(),
                                     "outputs");

    nvtx_profile_color("infer::infer_from_cpu", -2) {
        std::shared_ptr<buffer> context = create_buffer(batch_size);
        context->input_host_to_device(inputs_cpu, stream);
        (*this)(*context, stream);
        context->output_device_to_host(outputs_cpu, stream);
    }
}

void infer::infer_from_cpu(int batch_size,
                           const std::map<std::string, const void*>& inputs_cpu,
                           const std::map<std::string, void*>& outputs_cpu,
                           cudaStream_t stream) {
    nvtx_profile_color("infer::infer_from_cpu()", -2) {
        auto buffer = create_buffer(batch_size);
        buffer->input_host_to_device(inputs_cpu, stream);
        (*this)(*buffer, stream);
        buffer->output_device_to_host(outputs_cpu, stream);
    }
}

void infer::operator()(int batch_size,
                       const std::vector<const void*>& inputs_gpu,
                       const std::vector<void*>& outputs_gpu,
                       cudaStream_t stream) {
    if((int)inputs_gpu.size() != m->get_n_inputs())
        internal::throw_n_in_out_err(inputs_gpu.size(), m->get_n_inputs(),
                                     "inputs");
    if((int)outputs_gpu.size() != m->get_n_outputs())
        internal::throw_n_in_out_err(outputs_gpu.size(), m->get_n_outputs(),
                                     "outputs");

    nvtx_profile_color("infer::operator()", -2) {
        std::vector<void*> gpu_buffers;
        for(const void* p : inputs_gpu)
            gpu_buffers.push_back(const_cast<void*>(p));
        for(void* p : outputs_gpu)
            gpu_buffers.push_back(p);
        cuda_context->enqueue(batch_size, gpu_buffers.data(), stream, NULL);
    }
}

void infer::operator()(int batch_size,
                       const std::map<std::string, const void*>& inputs_gpu,
                       const std::map<std::string, void*>& outputs_gpu,
                       cudaStream_t stream) {
    nvtx_profile_color("infer::operator()", -2) {
        // concatenate bindings
        std::vector<std::pair<std::string, void*>> name_outputs;
        name_outputs.reserve(inputs_gpu.size() + outputs_gpu.size());
        for(auto it : inputs_gpu)
            name_outputs.push_back({it.first, const_cast<void*>(it.second)});
        name_outputs.insert(name_outputs.begin(), outputs_gpu.begin(),
                            outputs_gpu.end());

        // make bindings
        std::vector<void*> bindings(m->engine->getNbBindings(), nullptr);
        for(auto it : name_outputs) {
            const int bind_idx = m->engine->getBindingIndex(it.first.c_str());
            if(bind_idx != -1)
                bindings[bind_idx] = it.second;
            else if(debug)
                std::cerr << "[debug] ignored " << it.first
                          << ", because the NN doesn't have an output with "
                          << "the specified name" << std::endl;
        }

        // check if nothing is left behind
        for(unsigned i = 0; i < bindings.size(); ++i) {
            if(!bindings[i]) {
                auto msg = std::string(m->engine->getBindingName(i)) +
                           " is not specified in name_buffer_map";
                throw std::invalid_argument(msg);
            }
        }

        // run inference
        cuda_context->enqueue(batch_size, bindings.data(), stream, NULL);
    }
}

void infer::operator()(int batch_size, const std::vector<void*>& bindings,
                       cudaStream_t stream) {
    nvtx_profile_color("infer::operator()", -2) {
        cuda_context->enqueue(batch_size, (void**)bindings.data(), stream,
                              NULL);
    }
}
}
