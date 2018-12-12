/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void leaky_relu_kernel(const T* src_gpu, T* dst_gpu, int n_in,
                                      float slope) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;

        float val = src_gpu[idx];
        dst_gpu[idx] = max(val, slope * val);
    }

    template <typename T>
    void apply_leaky_relu(const T* src_gpu, T* dst_gpu, int n_in, float slope,
                          int batch_size, cudaStream_t stream) {
        int block_size = 1024;
        int grid_size = (int)std::ceil(1.0 * n_in * batch_size / block_size);
        dim3 grid(grid_size, batch_size);
        leaky_relu_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, dst_gpu, n_in * batch_size, slope);
    }

    // explicit instantiation (without this, link error will happen)
    template void apply_leaky_relu(const float*, float*, int, float, int,
                                   cudaStream_t);
    template void apply_leaky_relu(const __half*, __half*, int, float, int,
                                   cudaStream_t);
}
}
