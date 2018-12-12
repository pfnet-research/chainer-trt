/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include "plugins_base.hpp"

namespace chainer_trt {
namespace plugin {
    // Applies broadcast_to
    class broadcast_to : public plugin_base<broadcast_to> {
        nvinfer1::Dims in_dims, out_dims;
        int in_size, out_size;
        int *d_i_strides, *d_o_strides;

    public:
        broadcast_to(nvinfer1::Dims _in_dims, nvinfer1::Dims _out_dims);
        broadcast_to(const void* buf, size_t size);

        int initialize() override;
        void terminate() override;

        size_t getSerializationSize() override;
        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                 const nvinfer1::Dims* outputDims,
                                 int nbOutputs, nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format,
                                 int maxBatchSize) override;

        void serialize(void* buffer) override;

        nvinfer1::Dims get_in_dims() const { return in_dims; };
        nvinfer1::Dims get_out_dims() const { return out_dims; };

        const char* get_plugin_type() const override {
            return "chainer_trt_broadcast_to";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
