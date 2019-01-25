/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>

#include <NvInfer.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "include/chainer_trt_impl.hpp"

template <typename T>
std::vector<T> load_values(const std::string&);
template <>
std::vector<float> load_values(const std::string& filename);
template <>
std::vector<__half> load_values(const std::string& filename);

template <typename T>
std::vector<T> load_values_binary(const std::string&);
template <>
std::vector<float> load_values_binary(const std::string& filename);
template <>
std::vector<__half> load_values_binary(const std::string& filename);

template <typename T>
std::vector<T> repeat_array(const std::vector<T>& vec, int batch_size) {
    std::vector<T> out;
    for(int i = 0; i < batch_size; ++i)
        out.insert(out.end(), vec.begin(), vec.end());
    return out;
}

void assert_eq(float t1, float t2);
void assert_eq(int t1, int t2);

template <class T>
void assert_vector_eq(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for(unsigned i = 0; i < vec1.size(); ++i)
        assert_eq(vec1[i], vec2[i]);
}
template <>
void assert_vector_eq(const std::vector<__half>& vec1,
                      const std::vector<__half>& vec2);

void assert_vector_near(const std::vector<float>& vec1,
                        const std::vector<float>& vec2, float abs_error);
void assert_vector_near(const std::vector<__half>& vec1,
                        const std::vector<__half>& vec2, float abs_error);

void assert_dims_eq(const nvinfer1::Dims& dim1, const nvinfer1::Dims& dim2);

template <typename FloatType, typename PluginType>
void run_plugin_assert_core(PluginType& plugin_src, int batch_size,
                            const nvinfer1::Dims& in_dims,
                            const std::string& dir, float allowed_err = 0,
                            bool load_binary = false) {
    cudaSetDevice(0);

    std::vector<FloatType> in_vals, out_vals;
    if(load_binary) {
        in_vals = load_values_binary<FloatType>(dir + "/in.bin");
        out_vals = load_values_binary<FloatType>(dir + "/out.bin");
    } else {
        in_vals = load_values<FloatType>(dir + "/in.csv");
        out_vals = load_values<FloatType>(dir + "/out.csv");
    }

    const auto in_cpu = repeat_array(in_vals, batch_size);
    const auto expected_out_cpu = repeat_array(out_vals, batch_size);
    const int n_in = chainer_trt::internal::calc_n_elements(in_dims);

    // Make a plugin (and serialize, deserialize)
    std::unique_ptr<unsigned char[]> buf(
      new unsigned char[plugin_src.getSerializationSize()]);
    plugin_src.serialize(buf.get());
    PluginType plugin(buf.get(), plugin_src.getSerializationSize());
    plugin.initialize();

    const nvinfer1::Dims out_dims = plugin.getOutputDimensions(0, &in_dims, 1);
    const int n_out = chainer_trt::internal::calc_n_elements(out_dims);

    ASSERT_EQ(in_cpu.size(), n_in * batch_size);
    ASSERT_EQ(expected_out_cpu.size(), n_out * batch_size);

    // Prepare data
    FloatType *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(FloatType) * in_cpu.size());
    cudaMalloc((void**)&out_gpu, sizeof(FloatType) * expected_out_cpu.size());
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(FloatType) * in_cpu.size(),
               cudaMemcpyHostToDevice);

    // Run inference
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};
    plugin.enqueue(batch_size, ins_gpu, outs_gpu, NULL, 0);

    // Get result and assert
    std::vector<FloatType> out_cpu(expected_out_cpu.size());
    cudaMemcpy(out_cpu.data(), out_gpu, sizeof(FloatType) * out_cpu.size(),
               cudaMemcpyDeviceToHost);

    if(allowed_err < 1e-3)
        assert_vector_eq<FloatType>(out_cpu, expected_out_cpu);
    else
        assert_vector_near(out_cpu, expected_out_cpu, allowed_err);

    cudaFree(in_gpu);
    cudaFree(out_gpu);
}

template <typename PluginType>
void run_plugin_assert(PluginType& plugin_src, int batch_size,
                       nvinfer1::DataType data_type,
                       const nvinfer1::Dims& in_dims, const std::string& dir,
                       float allowed_err = 0, bool load_binary = false) {
    const nvinfer1::Dims out_dim =
      plugin_src.getOutputDimensions(0, &in_dims, 1);

    plugin_src.configureWithFormat(&in_dims, 1, &out_dim, 1, data_type,
                                   nvinfer1::PluginFormat::kNCHW, batch_size);

    if(data_type == nvinfer1::DataType::kFLOAT)
        run_plugin_assert_core<float>(plugin_src, batch_size, in_dims, dir,
                                      allowed_err, load_binary);
    else if(data_type == nvinfer1::DataType::kHALF)
        run_plugin_assert_core<__half>(plugin_src, batch_size, in_dims, dir,
                                       allowed_err, load_binary);
}

template <typename T>
void write_vector_to_file(const std::string& fn, const std::vector<T>& values) {
    std::ofstream ofs(fn);
    for(unsigned i = 0; i < values.size(); ++i) {
        if(i != 0)
            ofs << ",";
        ofs << values[i];
    }
    ofs << std::endl;
}

class BS : public chainer_trt::calibration_stream {
    int n_batch;
    std::mt19937 mt;

public:
    BS(int _n_batch) : n_batch(_n_batch) {}

    virtual int get_n_batch() override { return n_batch; }
    virtual int get_n_input() override { return 1; }

    virtual void get_batch(int i_batch, int input_idx,
                           const std::vector<int>& dims,
                           void* dst_buf_cpu) override {
        (void)i_batch;
        (void)input_idx;

        std::uniform_real_distribution<> rng(0.0, 1.0);
        const unsigned n_elements =
          chainer_trt::internal::calc_n_elements(dims);
        for(unsigned i = 0; i < n_elements; ++i)
            ((float*)dst_buf_cpu)[i] = rng(mt);
    }
};
