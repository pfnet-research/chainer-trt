/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../include/cuda/cuda_kernels.hpp"
#include "../include/plugins/resize.hpp"

namespace chainer_trt {
namespace plugin {

    resize::resize(int _n_channels, int _in_h, int _in_w, int _out_h,
                   int _out_w)
      : n_channels(_n_channels), in_h(_in_h), in_w(_in_w), out_h(_out_h),
        out_w(_out_w) {}

    resize::resize(const void *buf, size_t size) {
        (void)size;

        auto p = static_cast<const resize *>(buf);
        data_type = p->data_type;
        n_channels = p->n_channels;
        in_h = p->in_h;
        in_w = p->in_w;
        out_h = p->out_h;
        out_w = p->out_w;
    };

    nvinfer1::Dims resize::getOutputDimensions(int index,
                                               const nvinfer1::Dims *inputs,
                                               int nbInputDims) {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        return nvinfer1::DimsCHW(n_channels, out_h, out_w);
    }

    int resize::initialize() { return 0; }

    void resize::terminate() {}

    int resize::enqueue(int batchSize, const void *const *inputs,
                        void **outputs, void *workspace, cudaStream_t stream) {
        (void)workspace;

        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_resize(static_cast<const float *>(inputs[0]),
                             static_cast<float *>(outputs[0]), batchSize,
                             n_channels, in_h, in_w, out_h, out_w, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_resize(static_cast<const __half *>(inputs[0]),
                             static_cast<__half *>(outputs[0]), batchSize,
                             n_channels, in_h, in_w, out_h, out_w, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to resize::enqueue");
        }
        return 0;
    }

    void resize::serialize(void *buf) {
        auto p = static_cast<resize *>(buf);
        p->data_type = this->data_type;
        p->n_channels = this->n_channels;
        p->in_h = this->in_h;
        p->in_w = this->in_w;
        p->out_h = this->out_h;
        p->out_w = this->out_w;
    }
}
}
