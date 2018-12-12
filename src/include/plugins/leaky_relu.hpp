/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include "plugins_base.hpp"

namespace chainer_trt {
namespace plugin {
    class leaky_relu : public plugin_base<leaky_relu> {
        nvinfer1::Dims input_dims;
        float slope;

    public:
        leaky_relu(nvinfer1::Dims _dims, float _slope);
        leaky_relu(const void* buf, size_t size);

        size_t getSerializationSize() override;
        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream) override;

        void serialize(void* buffer) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_leaky_relu";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
