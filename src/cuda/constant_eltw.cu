/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void eltw_sum_kernel(const T* src_gpu, int n_in,
                                    const T* vals_gpu, int n_values,
                                    T* dst_gpu) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;
        const int batch = blockIdx.y;
        const int idx_in_vals = (n_values == 1 ? 0 : idx);
        dst_gpu[batch * n_in + idx] =
          vals_gpu[idx_in_vals] + src_gpu[batch * n_in + idx];
    }

    template <typename T>
    void apply_eltw_sum(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_in / block_size);
        dim3 grid(grid_size, batch_size);
        eltw_sum_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, n_in, vals_gpu, n_values, dst_gpu);
    }

    template <typename T>
    __global__ void eltw_sub_kernel(const T* src_gpu, int n_in,
                                    const T* vals_gpu, int n_values,
                                    T* dst_gpu) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;
        const int batch = blockIdx.y;
        const int idx_in_vals = (n_values == 1 ? 0 : idx);
        dst_gpu[batch * n_in + idx] =
          vals_gpu[idx_in_vals] - src_gpu[batch * n_in + idx];
    }

    template <typename T>
    void apply_eltw_sub(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_in / block_size);
        dim3 grid(grid_size, batch_size);
        eltw_sub_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, n_in, vals_gpu, n_values, dst_gpu);
    }

    template <typename T>
    __global__ void eltw_mul_kernel(const T* src_gpu, int n_in,
                                    const T* vals_gpu, int n_values,
                                    T* dst_gpu) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;
        const int batch = blockIdx.y;
        const int idx_in_vals = (n_values == 1 ? 0 : idx);
        dst_gpu[batch * n_in + idx] =
          vals_gpu[idx_in_vals] * src_gpu[batch * n_in + idx];
    }

    template <typename T>
    void apply_eltw_mul(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_in / block_size);
        dim3 grid(grid_size, batch_size);
        eltw_mul_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, n_in, vals_gpu, n_values, dst_gpu);
    }

    template <typename T>
    __global__ void eltw_div_kernel(const T* src_gpu, int n_in,
                                    const T* vals_gpu, int n_values,
                                    T* dst_gpu) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;
        const int batch = blockIdx.y;
        const int idx_in_vals = (n_values == 1 ? 0 : idx);
        dst_gpu[batch * n_in + idx] =
          vals_gpu[idx_in_vals] / src_gpu[batch * n_in + idx];
    }

    template <typename T>
    void apply_eltw_div(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_in / block_size);
        dim3 grid(grid_size, batch_size);
        eltw_div_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, n_in, vals_gpu, n_values, dst_gpu);
    }

    // explicit instantiation (without this, link error will happen)
    template void apply_eltw_sum(const float*, int, const float*, int, float*,
                                 int, cudaStream_t);
    template void apply_eltw_sub(const float*, int, const float*, int, float*,
                                 int, cudaStream_t);
    template void apply_eltw_mul(const float*, int, const float*, int, float*,
                                 int, cudaStream_t);
    template void apply_eltw_div(const float*, int, const float*, int, float*,
                                 int, cudaStream_t);
    template void apply_eltw_sum(const __half*, int, const __half*, int,
                                 __half*, int, cudaStream_t);
    template void apply_eltw_sub(const __half*, int, const __half*, int,
                                 __half*, int, cudaStream_t);
    template void apply_eltw_mul(const __half*, int, const __half*, int,
                                 __half*, int, cudaStream_t);
    template void apply_eltw_div(const __half*, int, const __half*, int,
                                 __half*, int, cudaStream_t);
}
}
