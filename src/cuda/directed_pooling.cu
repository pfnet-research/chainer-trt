/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <builtin_types.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {

    __device__ __half max(const __half a, const __half b) {
        return (a > b) ? a : b;
    }

    __device__ float max(const float a, const float b) { return fmaxf(a, b); }

    template <typename T>
    __global__ void horizontal_directed_pooling(const int B, const int C,
                                                const int H, const int W,
                                                const bool rev,
                                                const T *__restrict__ src,
                                                T *__restrict__ dst) {
        // int target = threadIdx.x;  // w_index
        const int h_idx = blockIdx.z;
        const int c_idx = blockIdx.y;
        const int b_idx = blockIdx.x;
        const int offset = b_idx * (C * H * W) + c_idx * (H * W) + h_idx * W;

        assert(threadIdx.x < W);

        assert(W <= 1024);
        __shared__ float buf[1024];
        buf[threadIdx.x] = (float)src[offset + threadIdx.x];
        __syncthreads();

        int x = threadIdx.x;

        if(rev) {
            for(int s = 1; s < W; s *= 2) {
                if(x + s < W) {
                    buf[x] = max(buf[x], buf[x + s]);
                }
                __syncthreads();
            }
        } else {
            for(int s = 1; s < W; s *= 2) {
                if(x - s >= 0) {
                    buf[x] = max(buf[x], buf[x - s]);
                }
                __syncthreads();
            }
        }
        dst[offset + x] = (T)buf[x];
    }

    template <typename T>
    __global__ void
    vertical_directed_pooling(const int B, const int C, const int H,
                              const int W, const bool rev,
                              const T *__restrict__ src, T *__restrict__ dst) {
        int target = blockIdx.y * blockDim.y + threadIdx.y;
        const int w_idx = target % W;
        const int c_idx = target / W;
        const int b_idx = blockIdx.x;

        const int offset = b_idx * (C * H * W) + c_idx * (H * W) + w_idx;
        const int step = W;
        assert(offset < B * C * H * W);
        assert(offset + step * (H - 1) <= B * C * H * W);
        if(rev) {
            int prev = offset + step * (H - 1);
            dst[prev] = src[prev];
            for(int i = 1; i < H; ++i) {
                int cur = prev - step;
                dst[cur] = max(src[cur], dst[prev]);
                prev = cur;
            }
        } else {
            int prev = offset;
            dst[prev] = src[prev];
            for(int i = 1; i < H; ++i) {
                int cur = prev + step;
                dst[cur] = max(src[cur], dst[prev]);
                prev = cur;
            }
        }
    }

    template <typename T>
    void apply_directed_pooling(const T *src_gpu, int B, int C, int H, int W,
                                bool horizontal, bool rev, T *dst_gpu,
                                cudaStream_t stream) {
        if(horizontal) {
            dim3 block(W);
            dim3 grid(B, C, H);
            horizontal_directed_pooling<<<grid, block, 0, stream>>>(
              B, C, H, W, rev, src_gpu, dst_gpu);
        } else {
            dim3 block(1, W);
            dim3 grid(B, C * W / block.y);
            vertical_directed_pooling<<<grid, block, 0, stream>>>(
              B, C, H, W, rev, src_gpu, dst_gpu);
        }
    }

    template void apply_directed_pooling(const float *src_gpu, int B, int C,
                                         int H, int W, bool horizontal,
                                         bool rev, float *dst_gpu,
                                         cudaStream_t stream);
    template void apply_directed_pooling(const __half *src_gpu, int B, int C,
                                         int H, int W, bool horizontal,
                                         bool rev, __half *dst_gpu,
                                         cudaStream_t stream);
}
}
