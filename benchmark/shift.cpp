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
static void benchmark_shift_core(benchmark::State& state,
                                 nvinfer1::DataType type) {
    cudaSetDevice(0);

    const int c = state.range(0), in_w = state.range(1), in_h = state.range(2);
    const int k = state.range(3), d = state.range(4);
    const nvinfer1::Dims dims = make_dims(c, in_h, in_w);
    std::vector<T> in_cpu(c * in_w * in_h, T());

    float *in_gpu, *out_gpu;
    cudaMalloc((void**)&in_gpu, sizeof(T) * in_cpu.size());
    cudaMalloc((void**)&out_gpu, sizeof(T) * in_cpu.size());
    cudaMemcpy(in_gpu, in_cpu.data(), sizeof(T) * in_cpu.size(),
               cudaMemcpyHostToDevice);

    plugin::shift shift(dims, k, k, d, d);
    shift.configureWithFormat(&dims, 1, NULL, 1, type,
                              nvinfer1::PluginFormat::kNCHW, 1);
    void *ins_gpu[] = {in_gpu}, *outs_gpu[] = {out_gpu};

    // Main benchmark loop
    for(auto _ : state) {
        shift.enqueue(1, ins_gpu, outs_gpu, NULL, 0);
        cudaStreamSynchronize(0);
    }

    cudaFree(in_gpu);
    cudaFree(out_gpu);
}

static void benchmark_shift_float(benchmark::State& state) {
    benchmark_shift_core<float>(state, nvinfer1::DataType::kFLOAT);
}

static void benchmark_shift_half(benchmark::State& state) {
    benchmark_shift_core<__half>(state, nvinfer1::DataType::kHALF);
}

static void make_shift_args(benchmark::internal::Benchmark* b) {
    const std::vector<int> ds{1, 2, 3};
    const std::vector<int> ks{3, 5};
    const std::vector<int> in_ss{8, 32, 128, 512, 2048};
    for(int s : in_ss)
        for(int k : ks)
            for(int d : ds)
                b->Args({k * k, s, s, k, d});
}

static void make_shift_args_actual_examples(benchmark::internal::Benchmark* b) {
    // Some examples actually used for an image recognition task
    b->Args({32, 192, 320, 5, 1});
    b->Args({64, 192, 320, 5, 1});
    b->Args({64, 96, 160, 5, 1});
    b->Args({64, 96, 160, 3, 2});
    b->Args({128, 96, 160, 3, 4});
    b->Args({128, 96, 160, 3, 1});
    b->Args({192, 96, 160, 3, 1});
    b->Args({128, 48, 80, 3, 1});
    b->Args({128, 24, 40, 3, 1});
    b->Args({128, 12, 20, 3, 1});
}

BENCHMARK(benchmark_shift_float) // ->Unit(benchmark::kMicrosecond)
  ->Apply(make_shift_args)
  ->Apply(make_shift_args_actual_examples);

BENCHMARK(benchmark_shift_half) // ->Unit(benchmark::kMicrosecond)
  ->Apply(make_shift_args)
  ->Apply(make_shift_args_actual_examples);
