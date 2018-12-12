/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_fp16.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#define OPT_SPLIT_W 1
#define OPT_UNROLL_H 1

namespace chainer_trt {
namespace plugin {
    template <typename T>
    __global__ void resize_kernel(const T* __restrict__ src,
                                  T* __restrict__ dst, int in_h, int in_w,
                                  int out_h, int out_w) {
        // See also: Chainer's implementation of linear interpolated resize
        // https://github.com/chainer/chainer/blob/v3.4.0/chainer/functions/array/resize_images.py#L24

        // blockIdx.y is channel index
        src = &src[blockIdx.y * in_h * in_w];
        dst = &dst[blockIdx.y * out_h * out_w +
                   (blockIdx.x << OPT_UNROLL_H) * out_w + threadIdx.x];

        const float u_t = (float)(in_h - 1) / max(out_h - 1, 1);
        const float v_t = (float)(in_w - 1) / max(out_w - 1, 1);

        #pragma unroll
        for(unsigned int j = 0; j < (1 << OPT_UNROLL_H); j++) {
            const float u = u_t * ((blockIdx.x << OPT_UNROLL_H) + j);

            const unsigned int umin = min(max((int)u, 0), in_h - 2);
            const unsigned int umax = umin + 1;
            const float um1 = umax - u;
            const float um2 = u - umin;

            #pragma unroll
            for(unsigned int i = 0; i < (1 << OPT_SPLIT_W); i++) {
                T* _dst = &dst[j * out_w + (i * out_w >> OPT_SPLIT_W)];

                const float v =
                  v_t * (threadIdx.x + (i * out_w >> OPT_SPLIT_W));

                const unsigned int vmin = min(max((int)v, 0), in_w - 2);
                const unsigned int vmax = vmin + 1;
                const float vm1 = vmax - v;
                const float vm2 = v - vmin;

                *_dst = (T)(vm1 * um1) * src[umin * in_w + vmin] +
                        (T)(vm2 * um1) * src[umin * in_w + vmax] +
                        (T)(vm1 * um2) * src[umax * in_w + vmin] +
                        (T)(vm2 * um2) * src[umax * in_w + vmax];
            }
        }
    }

    template <typename T>
    __global__ void
    resize_fallback_kernel(const T* __restrict__ src, T* __restrict__ dst,
                           int n_c, int in_h, int in_w, int out_h, int out_w) {
        float u, v;
        unsigned int c, idx, umin, vmin, umax, vmax;

        // See also: Chainer's implementation of linear interpolated resize
        // https://github.com/chainer/chainer/blob/v3.4.0/chainer/functions/array/resize_images.py#L24
        const unsigned int chw = n_c * out_h * out_w;

        // blockIdx.y is batch index
        const unsigned int in_batch_ofst = blockIdx.y * n_c * in_h * in_w;
        const unsigned int out_batch_ofst = blockIdx.y * chw;

        float u_t = (float)(in_h - 1) / max(out_h - 1, 1);
        float v_t = (float)(in_w - 1) / max(out_w - 1, 1);

        for(unsigned int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
            dst_idx < chw; dst_idx += blockDim.x * gridDim.x) {
            c = dst_idx / (out_h * out_w);
            idx = dst_idx % (out_h * out_w);

            u = u_t * (idx / out_w);
            v = v_t * (idx % out_w);
            umin = min((unsigned int)u, in_h - 2);
            vmin = min((unsigned int)v, in_w - 2);
            umax = umin + 1;
            vmax = vmin + 1;

            const int offset = in_batch_ofst + c * in_h * in_w;

            dst[out_batch_ofst + dst_idx] =
              (T)((vmax - v) * (umax - u)) *
                src[offset + max(min(umin, in_h - 1), 0) * in_w +
                    max(min(vmin, in_w - 1), 0)] +
              (T)((v - vmin) * (umax - u)) *
                src[offset + max(min(umin, in_h - 1), 0) * in_w +
                    max(min(vmax, in_w - 1), 0)] +
              (T)((vmax - v) * (u - umin)) *
                src[offset + max(min(umax, in_h - 1), 0) * in_w +
                    max(min(vmin, in_w - 1), 0)] +
              (T)((v - vmin) * (u - umin)) *
                src[offset + max(min(umax, in_h - 1), 0) * in_w +
                    max(min(vmax, in_w - 1), 0)];
        }
    }

    template <typename T>
    void apply_resize(const T* src, T* dst, int batch_size, int n_c, int in_h,
                      int in_w, int out_h, int out_w, cudaStream_t stream) {
        if((out_w & ((1 << OPT_SPLIT_W) - 1)) == 0 &&
           (out_h & ((1 << OPT_SPLIT_W) - 1)) == 0 && 1 < in_h && 1 < in_w) {
            // adjust for large width
            const int block_size = out_w >> OPT_SPLIT_W;
            dim3 grid(out_h >> OPT_UNROLL_H, n_c * batch_size);
            resize_kernel<T><<<grid, block_size, 0, stream>>>(
              src, dst, in_h, in_w, out_h, out_w);
        } else {
            const int block_size = 512;
            const int grid_size = 64; // 8 * numSMs
            dim3 grid(grid_size, batch_size);
            resize_fallback_kernel<T><<<grid, block_size, 0, stream>>>(
              src, dst, n_c, in_h, in_w, out_h, out_w);
        }
    }

    template void apply_resize(const float*, float*, int, int, int, int, int,
                               int, cudaStream_t);
    template void apply_resize(const __half*, __half*, int, int, int, int, int,
                               int, cudaStream_t);
}
}
