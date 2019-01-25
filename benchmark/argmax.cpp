/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/plugins/argmax.hpp"

using namespace chainer_trt;
using namespace chainer_trt::internal;

template <typename T>
static void benchmark_argmax_core(benchmark::State& state,
                                  nvinfer1::DataType type) {
    cudaSetDevice(0);

    const int c = state.range(0), s = state.range(1);
    std::vector<T> in_cpu(c * s, T());

    T *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(T) * in_cpu.size());
    cudaMalloc((void**)&out_gpu, sizeof(T) * in_cpu.size());
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(T) * in_cpu.size(),
               cudaMemcpyHostToDevice);

    auto dim = make_dims(c, s, 1);
    plugin::argmax argmax(dim);
    argmax.configureWithFormat(&dim, 1, NULL, 1, type,
                               nvinfer1::PluginFormat::kNCHW, 1);
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};

    // Main benchmark loop
    for(auto _ : state) {
        argmax.enqueue(1, ins_gpu, outs_gpu, NULL, 0);
        cudaStreamSynchronize(0);
    }

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaStreamSynchronize(0);
}

static void benchmark_argmax_float(benchmark::State& state) {
    benchmark_argmax_core<float>(state, nvinfer1::DataType::kFLOAT);
}

static void benchmark_argmax_half(benchmark::State& state) {
    benchmark_argmax_core<__half>(state, nvinfer1::DataType::kHALF);
}

static void make_argmax_args(benchmark::internal::Benchmark* b) {
    const std::vector<int> in_ss{1024,   4096,    16384,  65536,
                                 262144, 1048576, 4194304};
    const std::vector<int> cs{8, 16, 32, 64};
    for(int s : in_ss)
        for(int c : cs)
            b->Args({c, s});
}

BENCHMARK(benchmark_argmax_float)->Apply(make_argmax_args);
BENCHMARK(benchmark_argmax_half)->Apply(make_argmax_args);
