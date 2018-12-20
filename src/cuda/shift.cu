/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    template <typename T, unsigned int KH, unsigned int KW>
    __global__ void shift_kernel(const T* __restrict__ x, const int c,
                                 const int h, const int dy, const int dx,
                                 T* __restrict__ y) {
        const unsigned int w = blockDim.x + 1024 * (gridDim.x - 1);
        const unsigned int hw = h * w;
        // blockIdx.z is batch index
        const unsigned int batch_offset = blockIdx.z * c * hw;

        int kx = 0;
        int ky = 0;
        {
            const unsigned int n_groups = KH * KW;
            const unsigned int group_size = c / n_groups;

            // blockIdx.y is channel index
            unsigned int group_idx = blockIdx.y / group_size;

            // Make sure that center group is last
            if(group_idx == (n_groups - 1) / 2)
                group_idx = n_groups - 1;
            else if(group_idx == n_groups - 1)
                group_idx = (n_groups - 1) / 2;

            if(group_idx < n_groups) {
                ky = (group_idx / KW) - KH / 2;
                kx = (group_idx % KW) - KW / 2;
            }
        }

        unsigned int offset = batch_offset + blockIdx.y * hw;
        // blockIdx.x is width index
        const int out_col = threadIdx.x + 1024 * blockIdx.x;
        int out_row = 0;

        y = &y[offset + out_col];

        int in_col = -kx * dx + out_col;
        if(in_col >= 0 && in_col < w) {
            offset += in_col;
            int in_row_offset = ky * dy;

            for(; out_row < in_row_offset; out_row++) {
                *y = 0;
                y += w;
            }

            unsigned int copy_h = min(h, h + in_row_offset);
            for(; out_row < copy_h; out_row++) {
                int in_row = -in_row_offset + out_row;
                *y = x[offset + in_row * w];
                y += w;
            }
        }
        for(; out_row < h; out_row++) {
            *y = 0;
            y += w;
        }
    }

    template <typename T>
    __global__ void
    shift_fallback_kernel(const T* __restrict__ x, const int c, const int h,
                          const int w, const int kh, const int kw, const int dy,
                          const int dx, T* __restrict__ y) {
        const unsigned int hw = h * w;
        const unsigned int chw = c * hw;
        // blockDim.y is batch index
        const unsigned int ofst = blockIdx.y * chw;
        const unsigned int n_groups = kh * kw;
        const unsigned int group_size = c / n_groups;
        const unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
        for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < chw;
            i += stride) {
            // Based on
            // https://github.com/chainer/chainer/pull/4041/files#diff-4cb6895e9a4acd09178c617ff177d0d0
            unsigned int b0 = i / chw;
            unsigned int rest = (i % chw);
            unsigned int c0 = rest / hw;
            rest %= hw;

            unsigned int out_row = rest / w;
            unsigned int out_col = rest % w;

            unsigned int group_idx = c0 / group_size;

            if(group_idx == (n_groups - 1) / 2)
                group_idx = n_groups - 1;
            else if(group_idx == n_groups - 1)
                group_idx = (n_groups - 1) / 2;

            unsigned int ky = (group_idx / kw) - kh / 2;
            unsigned int kx = (group_idx % kw) - kw / 2;
            if(group_idx >= n_groups) {
                ky = 0;
                kx = 0;
            }

            signed int in_row = -ky * dy + out_row;
            signed int in_col = -kx * dx + out_col;
            if(in_row >= 0 && in_row < h && in_col >= 0 && in_col < w)
                y[ofst + i] =
                  x[ofst + b0 * chw + c0 * hw + in_row * w + in_col];
            else
                y[ofst + i] = 0;
        }
    }

    template <typename T>
    void apply_shift(const T* src_gpu, int batch_size, int c, int h, int w,
                     int kh, int kw, int dy, int dx, int grid_size,
                     int block_size, T* dst_gpu, cudaStream_t stream) {
        if(w < 1024 || w % 1024 == 0) {
            dim3 grid(max(1, w / 1024), c, batch_size);
            if(kh == 7 && kw == 7) {
                shift_kernel<T, 7, 7><<<grid, min(w, 1024), 0, stream>>>(
                  src_gpu, c, h, dy, dx, dst_gpu);
                return;
            } else if(kh == 5 && kw == 5) {
                shift_kernel<T, 5, 5><<<grid, min(w, 1024), 0, stream>>>(
                  src_gpu, c, h, dy, dx, dst_gpu);
                return;
            } else if(kh == 3 && kw == 3) {
                shift_kernel<T, 3, 3><<<grid, min(w, 1024), 0, stream>>>(
                  src_gpu, c, h, dy, dx, dst_gpu);
                return;
            }
        }

        dim3 grid(grid_size, batch_size);
        shift_fallback_kernel<T><<<grid, block_size, 0, stream>>>(
          src_gpu, c, h, w, kh, kw, dy, dx, dst_gpu);
    }

    template void apply_shift(const float*, int, int, int, int, int, int, int,
                              int, int, int, float*, cudaStream_t);
    template void apply_shift(const __half*, int, int, int, int, int, int, int,
                              int, int, int, __half*, cudaStream_t);
}
}
