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
        static nvinfer1::ILayer*
        build_layer(network_def network, const picojson::object& layer_params,
                    nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                    const std::string& model_dir);

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
    };
}
}
