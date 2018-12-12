/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void where_kernel(const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 const T* __restrict__ c, T* __restrict__ dst,
                                 const unsigned int n_in) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_in <= idx)
            return;
        const int batch = blockIdx.y;
        dst[batch * n_in + idx] = (int)a[batch * n_in + idx]
                                    ? b[batch * n_in + idx]
                                    : c[batch * n_in + idx];
    }

    template <typename T>
    void apply_where(const T* a, const T* b, const T* c, T* dst, int n_in,
                     int batch_size, cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_in / block_size);
        dim3 grid(grid_size, batch_size);
        where_kernel<T><<<grid, block_size, 0, stream>>>(a, b, c, dst, n_in);
    }

    template void apply_where(const float*, const float*, const float*, float*,
                              int, int, cudaStream_t);
    template void apply_where(const __half*, const __half*, const __half*,
                              __half*, int, int, cudaStream_t);
}
}
