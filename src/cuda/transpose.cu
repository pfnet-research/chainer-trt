/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

namespace chainer_trt {
namespace plugin {
    __global__ void transpose_kernel(const float* d_src, float* d_dst,
                                     int* d_indexes, int in_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < in_size)
            d_dst[blockIdx.y * in_size + d_indexes[idx]] =
              d_src[blockIdx.y * in_size + idx];
    }

    __global__ void transpose_indexes(int* d_dst, int* i_strides, int* shuffle,
                                      int* i_d, int* o_strides, int id_size,
                                      int in_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // _h

        if(idx < in_size) {
            int out_idx = 0;
            for(int i = 0; i < id_size; i++)
                out_idx += (idx / i_strides[shuffle[i]] % i_d[shuffle[i]]) *
                           o_strides[i];

            d_dst[idx] = out_idx;
        }
    }

    void apply_transpose(const float* d_src, float* d_dst, int* d_indexes,
                         int in_size, int batch_size, cudaStream_t stream) {
        const int thread_size = 1024;
        const int block_size = (int)std::ceil(1.0 * in_size / thread_size);
        dim3 grid(block_size, batch_size);
        transpose_kernel<<<grid, thread_size, 0, stream>>>(d_src, d_dst,
                                                           d_indexes, in_size);
    }

    void initialize_transpose_indexes(int* d_dst, int* i_strides, int* shuffle,
                                      int* i_d, int* o_strides, int in_size,
                                      int id_size) {
        const int thread_size = 1024;
        const int block_size = (int)std::ceil(1.0 * in_size / thread_size);
        transpose_indexes<<<block_size, thread_size>>>(
          d_dst, i_strides, shuffle, i_d, o_strides, id_size, in_size);
    }
}
}
