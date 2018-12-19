/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

template <typename T>
__global__ void increment_kernel(const T* in, T* out, int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(n <= idx)
        return;

    out[idx] = in[idx] + T(1);
}

template <typename T>
void increment(const T* in, T* out, int n, cudaStream_t stream) {
    const int thread_size = 1024;
    const int block_size = (int)std::ceil(1.0 * thread_size);
    increment_kernel<<<block_size, thread_size, 0, stream>>>(in, out, n);
}

template void increment(const float*, float*, int, cudaStream_t);
template void increment(const __half*, __half*, int, cudaStream_t);
