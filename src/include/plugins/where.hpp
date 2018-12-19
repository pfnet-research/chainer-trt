/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <chainer_trt/plugin.hpp>

namespace chainer_trt {
namespace plugin {
    class where : public plugin_base<where> {
        nvinfer1::Dims input_dims;

    public:
        where(nvinfer1::Dims dims);
        where(const void* buf, size_t size);

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        size_t getSerializationSize() override;

        void serialize(void* buffer) override;

        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override {
            (void)index;
            (void)inputs;
            (void)nbInputDims;
            return inputs[0];
        }

        void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                 const nvinfer1::Dims* outputDims,
                                 int nbOutputs, nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format,
                                 int maxBatchSize) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_where";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
