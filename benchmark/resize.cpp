/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>

#include "chainer_trt/chainer_trt.hpp"

#include "../src/include/chainer_trt_impl.hpp"

using namespace chainer_trt;
using namespace chainer_trt::internal;

template <typename T>
static void benchmark_resize_core(benchmark::State& state,
                                  nvinfer1::DataType type) {
    cudaSetDevice(0);

    const int n_channels = state.range(0);
    const int in_w = state.range(1);
    const int in_h = state.range(2);
    const int out_w = state.range(3);
    const int out_h = state.range(4);
    const int n_in = n_channels * in_w * in_h;
    const int n_out = n_channels * out_w * out_h;
    std::vector<T> in_cpu(n_in, T());

    float *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(T) * n_in);
    cudaMalloc((void**)&out_gpu, sizeof(T) * n_out);
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(T) * n_in, cudaMemcpyHostToDevice);

    const auto in_dims = make_dims(n_channels, in_h, in_w);
    plugin::resize resize(n_channels, in_h, in_w, out_h, out_w);
    resize.configureWithFormat(&in_dims, 1, NULL, 1, type,
                               nvinfer1::PluginFormat::kNCHW, 1);
    resize.initialize();
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};

    // Main benchmark loop
    for(auto _ : state) {
        resize.enqueue(1, ins_gpu, outs_gpu, NULL, 0);
        cudaStreamSynchronize(0);
    }

    cudaFree(in_gpu);
    cudaFree(out_gpu);
}

static void benchmark_resize_float(benchmark::State& state) {
    benchmark_resize_core<float>(state, nvinfer1::DataType::kFLOAT);
}

static void benchmark_resize_half(benchmark::State& state) {
    benchmark_resize_core<__half>(state, nvinfer1::DataType::kHALF);
}

static void
make_resize_args_square_integer_ratio(benchmark::internal::Benchmark* b) {
    const std::vector<int> ss{8, 32, 128, 512, 2048};
    for(int out_s : ss)
        for(int in_s : ss)
            b->Args({1, in_s, in_s, out_s, out_s});
}

BENCHMARK(benchmark_resize_float)->Apply(make_resize_args_square_integer_ratio);
BENCHMARK(benchmark_resize_half)->Apply(make_resize_args_square_integer_ratio);
