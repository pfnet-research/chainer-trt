/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <chainer_trt/plugin.hpp>

namespace chainer_trt {
namespace plugin {
    class directed_pooling : public plugin_base<directed_pooling> {
        int C, H, W;
        int horizontal, rev;

    public:
        directed_pooling(nvinfer1::Dims in_dim, int horizontal_, int rev_);
        directed_pooling(const void* buf, size_t size);

        static nvinfer1::ILayer*
        build_layer(network_def network, const picojson::object& layer_params,
                    nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                    const std::string& model_dir);

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        size_t getSerializationSize() override {
            return sizeof(directed_pooling);
        }

        void serialize(void* buffer) override;

        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;
    };
}
}
