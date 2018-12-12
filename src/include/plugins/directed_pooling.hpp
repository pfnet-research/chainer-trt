/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include "plugins_base.hpp"

namespace chainer_trt {
namespace plugin {
    class directed_pooling : public plugin_base<directed_pooling> {
        int C, H, W;
        int horizontal, rev;

    public:
        directed_pooling(nvinfer1::Dims in_dim, int horizontal_, int rev_);
        directed_pooling(const void* buf, size_t size);

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        size_t getSerializationSize() override {
            return sizeof(directed_pooling);
        }

        void serialize(void* buffer) override;

        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_directed_pooling";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
