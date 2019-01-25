/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <stdexcept>

#include <cuda_fp16.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#define OPT_SPLIT_W 3
#define OPT_UNROLL_H 1

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void resize_argmax_kernel(const T* __restrict__ src,
                                         T* __restrict__ dst, int n_c, int in_h,
                                         int in_w, int out_h, int out_w) {
        // blockIdx.z is batch index
        src = &src[blockIdx.z * n_c * in_h * in_w];

        const float u_t = (float)(in_h - 1) / max(out_h - 1, 1);
        const float v_t = (float)(in_w - 1) / max(out_w - 1, 1);

        unsigned int max_idx[1 << OPT_UNROLL_H] = {0};
        float max_val[1 << OPT_UNROLL_H];

#pragma unroll
        for(unsigned int j = 0; j < (1 << OPT_UNROLL_H); j++)
            max_val[j] = -INFINITY;

        // blockIdx.y is partition index of x direction
        const unsigned int x =
          (blockIdx.y * out_w >> OPT_SPLIT_W) + threadIdx.x;

        // blockIdx.x is y index
        const unsigned int y = blockIdx.x << OPT_UNROLL_H;

        const float v = v_t * x;
        const unsigned int vmin = min(max((int)v, 0), in_w - 2);
        const unsigned int vmax = vmin + 1;
        const float vm1 = vmax - v;
        const float vm2 = v - vmin;

        dst = &dst[blockIdx.z * out_h * out_w + y * out_w + x];

        for(unsigned int j = 0; j < (1 << OPT_UNROLL_H); j++) {
            const float u = u_t * (y + j);
            const unsigned int umin = min(max((int)u, 0), in_h - 2);
            const unsigned int umax = umin + 1;
            const float um1 = umax - u;
            const float um2 = u - umin;

            for(unsigned int c = 0; c < n_c; c++) {
                const T* _src = &src[c * in_h * in_w];

                const float val =
                  (vm1 * um1) * (float)_src[umin * in_w + vmin] +
                  (vm2 * um1) * (float)_src[umin * in_w + vmax] +
                  (vm1 * um2) * (float)_src[umax * in_w + vmin] +
                  (vm2 * um2) * (float)_src[umax * in_w + vmax];

                if(val > max_val[j]) {
                    max_idx[j] = c;
                    max_val[j] = val;
                }
            }
        }

#pragma unroll
        for(unsigned int j = 0; j < (1 << OPT_UNROLL_H); j++) {
            *dst = (T)max_idx[j];
            dst += out_w;
        }
    }

    bool is_supported_size_by_resize_argmax(int in_h, int in_w, int out_h,
                                            int out_w) {
        if(in_h <= 1 || in_w <= 1)
            return false;
        if((out_w & ((1 << OPT_SPLIT_W) - 1)) != 0 ||
           (out_h & ((1 << OPT_SPLIT_W) - 1)) != 0)
            return false;
        return true;
    }

    template <typename T>
    void apply_resize_argmax(const T* src, T* dst, int batch_size, int n_c,
                             int in_h, int in_w, int out_h, int out_w,
                             cudaStream_t stream) {
        if(!is_supported_size_by_resize_argmax(in_h, in_w, out_h, out_w))
            throw std::range_error("input/output size is not supported");

        int block_size = out_w >> OPT_SPLIT_W; // adjust for large width
        dim3 grid(out_h >> OPT_UNROLL_H, 1 << OPT_SPLIT_W, batch_size);
        resize_argmax_kernel<<<grid, block_size, 0, stream>>>(
          src, dst, n_c, in_h, in_w, out_h, out_w);
    }

    template void apply_resize_argmax(const float*, float*, int, int, int, int,
                                      int, int, cudaStream_t);
    template void apply_resize_argmax(const __half*, __half*, int, int, int,
                                      int, int, int, cudaStream_t);
}
}
