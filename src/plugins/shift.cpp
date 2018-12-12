/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../include/cuda/cuda_kernels.hpp"
#include "../include/plugins/shift.hpp"

namespace chainer_trt {
namespace plugin {

    shift::shift(nvinfer1::Dims dims, int _kw, int _kh, int _dx, int _dy)
      : kw(_kw), kh(_kh), dx(_dx), dy(_dy), c(dims.d[0]), h(dims.d[1]),
        w(dims.d[2]) {
        this->setLaunchConfiguration();
    }

    shift::shift(const void* buf, size_t size) {
        (void)size;
        assert(size == sizeof(shift));

        auto p = static_cast<const shift*>(buf);
        data_type = p->data_type;
        kw = p->kw;
        kh = p->kh;
        dx = p->dx;
        dy = p->dy;
        c = p->c;
        h = p->h;
        w = p->w;
        setLaunchConfiguration();
    }

    void shift::setLaunchConfiguration() {
        this->block_size = 512;
        int device;
        cudaGetDevice(&device);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
        this->grid_size = 8 * numSMs;
    }

    void shift::serialize(void* buffer) {
        auto p = static_cast<shift*>(buffer);
        p->data_type = data_type;
        p->kw = kw;
        p->kh = kh;
        p->dx = dx;
        p->dy = dy;
        p->c = c;
        p->h = h;
        p->w = w;
    }

    nvinfer1::Dims shift::getOutputDimensions(int index,
                                              const nvinfer1::Dims* inputs,
                                              int nbInputDims) {
        (void)index;
        (void)nbInputDims;

        return inputs[0];
    }

    int shift::enqueue(int batchSize, const void* const* inputs, void** outputs,
                       void* workspace, cudaStream_t stream) {
        (void)workspace;

        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_shift<float>((const float*)inputs[0], batchSize, c, h, w,
                                   kh, kw, dy, dx, grid_size, block_size,
                                   (float*)outputs[0], stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_shift<__half>((const __half*)inputs[0], batchSize, c, h,
                                    w, kh, kw, dy, dx, grid_size, block_size,
                                    (__half*)outputs[0], stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to shift::enqueue");
        }
        return 0;
    }
}
}
