/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void slice_kernel(const T* src_gpu, T* dest_gpu,
                                 int* mapping_gpu, int n_src, int n_dst) {
        const int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(n_dst <= dst_idx)
            return;
        const int src_idx = mapping_gpu[dst_idx];
        dest_gpu[blockIdx.y * n_dst + dst_idx] =
          src_gpu[blockIdx.y * n_src + src_idx]; // blockIdx.y is batch idx
    }

    template <typename T>
    void apply_slice(const T* src_gpu, T* dest_gpu, int* mapping_gpu, int n_src,
                     int n_dst, int batch_size, cudaStream_t stream) {
        const int block_size = 1024;
        const int grid_size = (int)std::ceil(1.0 * n_dst / block_size);
        dim3 grid(grid_size, batch_size);
        slice_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, dest_gpu, mapping_gpu, n_src, n_dst);
    }

    template void apply_slice(const float*, float*, int*, int, int, int,
                              cudaStream_t);
    template void apply_slice(const __half*, __half*, int*, int, int, int,
                              cudaStream_t);
}
}
