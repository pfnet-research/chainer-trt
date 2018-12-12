/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void broadcast_to_kernel(const T *d_src, T *d_dst,
                                        int *d_i_strides, int *d_o_strides,
                                        int in_size, int out_size,
                                        int nb_dims) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < out_size) {
            // calc offset relationship between input & output
            int in_idx = 0;
            int f = idx;
            for(int i = 0; i < nb_dims; i++) {
                in_idx += (f / d_o_strides[i]) * d_i_strides[i];
                f = f % d_o_strides[i];
            }
            d_dst[blockIdx.y * out_size + idx] =
              d_src[blockIdx.y * in_size + in_idx];
        }
    }

    template <typename T>
    void apply_broadcast_to(const T *d_src, T *d_dst, int *d_i_strides,
                            int *d_o_strides, int in_size, int out_size,
                            int nb_dims, int batch_size, cudaStream_t stream) {
        const int thread_size = 1024;
        const int block_size = (int)std::ceil(1.0 * out_size / thread_size);
        dim3 grid(block_size, batch_size);
        broadcast_to_kernel<<<grid, thread_size, 0, stream>>>(
          d_src, d_dst, d_i_strides, d_o_strides, in_size, out_size, nb_dims);
    }

    template void apply_broadcast_to(const float *, float *, int *, int *, int,
                                     int, int, int, cudaStream_t);
    template void apply_broadcast_to(const __half *, __half *, int *, int *,
                                     int, int, int, int, cudaStream_t);
}
}
