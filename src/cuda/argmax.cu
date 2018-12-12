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
    __global__ void
    argmax_kernel(const T* __restrict__ d_src, T* __restrict__ d_dst,
                  const unsigned int channel, const unsigned int stride,
                  const unsigned int in_size) {

        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(stride <= idx)
            return;

        // blockIdx.y is batch idx
        const unsigned int ofst_src = in_size * blockIdx.y;
        // here stride is equivalent to out size
        const unsigned int ofst_dst = stride * blockIdx.y;

        unsigned int max_idx = 0;
        T max_val = d_src[ofst_src + idx];

        for(unsigned int i = 1; i < channel; i++) {
            const T val = d_src[ofst_src + i * stride + idx];
            if(val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        d_dst[ofst_dst + idx] = (T)max_idx;
    }

    template <typename T>
    void apply_argmax(const T* d_src, T* d_dst, int channel, int stride,
                      int in_size, int batch_size, cudaStream_t stream) {
        const int thread_size = 1024;
        const int block_size = (int)std::ceil(1.0 * stride / thread_size);
        dim3 grid(block_size, batch_size);
        argmax_kernel<<<grid, thread_size, 0, stream>>>(d_src, d_dst, channel,
                                                        stride, in_size);
    }

    template void apply_argmax(const float*, float*, int, int, int, int,
                               cudaStream_t);
    template void apply_argmax(const __half*, __half*, int, int, int, int,
                               cudaStream_t);
}
}
