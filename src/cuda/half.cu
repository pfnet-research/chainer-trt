/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

// This is a utility for conversion between float and half.
// From CUDA 9.2, __float2half/__half2float can be called from host and GCC can
// compile them, but with CUDA 9.0 they can't, so this utils are needed.

#include <cuda_fp16.h>
#include <iostream>

namespace chainer_trt {
namespace internal {

    __global__ void float2half_kernel(const float* src, __half* dst, int n) {
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n <= idx)
            return;
        dst[idx] = __float2half(src[idx]);
    }

    __global__ void half2float_kernel(const __half* src, float* dst, int n) {
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n <= idx)
            return;
        dst[idx] = __float2half(src[idx]);
    }

    void float2half(const float* src, __half* dst, int n) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n / block_size);
        float* src_g = NULL;
        __half* dst_g = NULL;
        cudaMalloc(&src_g, sizeof(float) * n);
        cudaMalloc(&dst_g, sizeof(__half) * n);
        cudaMemcpy(src_g, src, sizeof(float) * n, cudaMemcpyHostToDevice);
        float2half_kernel<<<grid_size, block_size, 0, 0>>>(src_g, dst_g, n);
        cudaMemcpy(dst, dst_g, sizeof(__half) * n, cudaMemcpyDeviceToHost);
        cudaFree(src_g);
        cudaFree(dst_g);
    }

    void half2float(const __half* src, float* dst, int n) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n / block_size);
        __half* src_g = NULL;
        float* dst_g = NULL;
        cudaMalloc(&src_g, sizeof(__half) * n);
        cudaMalloc(&dst_g, sizeof(float) * n);
        cudaMemcpy(src_g, src, sizeof(__half) * n, cudaMemcpyHostToDevice);
        half2float_kernel<<<grid_size, block_size, 0, 0>>>(src_g, dst_g, n);
        cudaMemcpy(dst, dst_g, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaFree(src_g);
        cudaFree(dst_g);
    }
}
}
