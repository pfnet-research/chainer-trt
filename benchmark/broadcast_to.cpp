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
static void benchmark_broadcast_to_core(benchmark::State& state,
                                        nvinfer1::DataType type) {
    cudaSetDevice(0);

    const int in_ch = state.range(0);
    const int in_h = state.range(1);
    const int in_w = state.range(2);
    const int out_ch = state.range(3);
    const int out_h = state.range(4);
    const int out_w = state.range(5);
    const int n_in = in_ch * in_h * in_w;
    const int n_out = out_ch * out_h * out_w;

    std::vector<T> in_cpu(n_in, T());

    float *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(T) * n_in);
    cudaMalloc((void**)&out_gpu, sizeof(T) * n_out);
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(T) * n_in, cudaMemcpyHostToDevice);

    auto in_dim = make_dims(in_ch, in_h, in_w);
    auto out_dim = make_dims(out_ch, out_h, out_w);
    plugin::broadcast_to broadcast_to(in_dim, out_dim);
    broadcast_to.configureWithFormat(&in_dim, 1, &out_dim, 1, type,
                                     nvinfer1::PluginFormat::kNCHW, 1);
    broadcast_to.initialize();
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};

    // Main benchmark loop
    for(auto _ : state) {
        broadcast_to.enqueue(1, ins_gpu, outs_gpu, NULL, 0);
        cudaStreamSynchronize(0);
    }

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaStreamSynchronize(0);
}

static void benchmark_broadcast_to_float(benchmark::State& state) {
    benchmark_broadcast_to_core<float>(state, nvinfer1::DataType::kFLOAT);
}

static void benchmark_broadcast_to_half(benchmark::State& state) {
    benchmark_broadcast_to_core<__half>(state, nvinfer1::DataType::kHALF);
}

static void make_broadcast_to_args(benchmark::internal::Benchmark* b) {
    const std::vector<int> out_ch{64, 1024};
    const std::vector<int> out_hw{64, 1024};

    for(int ch : out_ch) {
        for(int hw : out_hw) {
            b->Args({1, 1, 1, ch, hw, hw});
            b->Args({ch, 1, 1, ch, hw, hw});
            b->Args({1, hw, 1, ch, hw, hw});
            b->Args({1, 1, hw, ch, hw, hw});
            b->Args({ch, hw, 1, ch, hw, hw});
            b->Args({1, hw, hw, ch, hw, hw});
            b->Args({ch, 1, hw, ch, hw, hw});
            b->Args({ch, hw, hw, ch, hw, hw});
        }
    }
}

BENCHMARK(benchmark_broadcast_to_float)->Apply(make_broadcast_to_args);
BENCHMARK(benchmark_broadcast_to_half)->Apply(make_broadcast_to_args);
