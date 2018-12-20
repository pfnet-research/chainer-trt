/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/plugins/directed_pooling.hpp"

using namespace chainer_trt;
using namespace chainer_trt::internal;

template <typename T>
static void benchmark_directed_pooling_core(benchmark::State& state,
                                            nvinfer1::DataType type) {
    cudaSetDevice(0);

    const int c = state.range(0), in_w = state.range(1), in_h = state.range(2);
    const int horizontal = state.range(3);
    const int rev = state.range(4);
    const nvinfer1::Dims dims = make_dims(c, in_h, in_w);
    std::vector<T> in_cpu(c * in_w * in_h, T(1.0));

    T *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(T) * in_cpu.size());
    cudaMalloc((void**)&out_gpu, sizeof(T) * in_cpu.size());
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(T) * in_cpu.size(),
               cudaMemcpyHostToDevice);

    plugin::directed_pooling dp(dims, horizontal, rev);
    dp.configureWithFormat(&dims, 1, NULL, 1, type,
                           nvinfer1::PluginFormat::kNCHW, 1);
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};

    // Main benchmark loop
    for(auto _ : state) {
        dp.enqueue(1, ins_gpu, outs_gpu, NULL, 0);
        cudaStreamSynchronize(0);
    }

    cudaFree(in_gpu);
    cudaFree(out_gpu);
}

static void benchmark_directed_pooling_float(benchmark::State& state) {
    benchmark_directed_pooling_core<float>(state, nvinfer1::DataType::kFLOAT);
}

static void benchmark_directed_pooling_half(benchmark::State& state) {
    benchmark_directed_pooling_core<__half>(state, nvinfer1::DataType::kHALF);
}

static void make_directed_pooling_args(benchmark::internal::Benchmark* b) {
    b->Args({128, 160, 80, 0, 0});
    b->Args({128, 160, 80, 0, 1});
    b->Args({128, 160, 80, 1, 0});
    b->Args({128, 160, 80, 1, 1});
}

BENCHMARK(benchmark_directed_pooling_float) // ->Unit(benchmark::kMicrosecond)
  ->Apply(make_directed_pooling_args);
BENCHMARK(benchmark_directed_pooling_half) // ->Unit(benchmark::kMicrosecond)
  ->Apply(make_directed_pooling_args);
