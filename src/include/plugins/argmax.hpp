/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include "plugins_base.hpp"

namespace chainer_trt {
namespace plugin {
    // Applies argmax to channel (1st dim) of input
    class argmax : public plugin_base<argmax> {
        nvinfer1::Dims input_dims;

    public:
        argmax(nvinfer1::Dims _dims);
        argmax(const void* buf, size_t size);

        size_t getSerializationSize() override;
        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream) override;

        void serialize(void* buffer) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_argmax";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
