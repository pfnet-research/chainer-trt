/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <chainer_trt/plugin.hpp>

namespace chainer_trt {
namespace plugin {
    class constant_elementwise : public plugin_base<constant_elementwise> {
        nvinfer1::Dims dims;
        nvinfer1::ElementWiseOperation op;

        size_t n_in; // = dims.d[0] * dims.d[1] * ... * dims.d[dims.nbDims-1]
        std::vector<float> values;
        float* values_gpu;

    public:
        constant_elementwise(nvinfer1::Dims _dims,
                             nvinfer1::ElementWiseOperation _op,
                             const std::vector<float>& _values);

        constant_elementwise(const void* buf, size_t size);

        static nvinfer1::ILayer*
        build_layer(network_def network, const picojson::object& layer_params,
                    nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                    const std::string& model_dir);

        void terminate() override;

        int initialize() override;

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

            return dims;
        }

        const char* get_plugin_type() const override {
            return "chainer_trt_constant_elementwise";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }

        // helpers for test
        nvinfer1::ElementWiseOperation get_op() const { return op; }
        std::vector<float> get_values() const { return values; }
        nvinfer1::Dims get_dims() const { return dims; }
    };
}
}
