/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_runtime.h>

#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/leaky_relu.hpp"
#include <chainer_trt/chainer_trt.hpp>
#include <chainer_trt/external/picojson_helper.hpp>

namespace chainer_trt {
namespace plugin {

    leaky_relu::leaky_relu(nvinfer1::Dims _dims, float _slope)
      : input_dims(_dims), slope(_slope) {}

    leaky_relu::leaky_relu(const void* buf, size_t size) {
        (void)size;
        auto p = static_cast<const leaky_relu*>(buf);
        input_dims = p->input_dims;
        data_type = p->data_type;
        slope = p->slope;
    }

    nvinfer1::ILayer* leaky_relu::build_layer(
      network_def network, const picojson::object& layer_params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names,
      const std::string& model_dir) {
        (void)dt;
        (void)model_dir;

        const auto source = param_get<std::string>(layer_params, "source");
        const float slope = param_get<double>(layer_params, "slope");
        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        auto p = new plugin::leaky_relu(input->getDimensions(), slope);
        return network->addPluginExt(&input, 1, *p);
    }

    size_t leaky_relu::getSerializationSize() { return sizeof(leaky_relu); }

    nvinfer1::Dims leaky_relu::getOutputDimensions(int index,
                                                   const nvinfer1::Dims* inputs,
                                                   int nbInputDims) {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        return input_dims;
    }

    void leaky_relu::serialize(void* buffer) {
        auto p = static_cast<leaky_relu*>(buffer);
        p->input_dims = input_dims;
        p->data_type = data_type;
        p->slope = slope;
    }

    int leaky_relu::enqueue(int batchSize, const void* const* inputs,
                            void** outputs, void* workspace,
                            cudaStream_t stream) {
        (void)workspace;

        const int n_total = internal::calc_n_elements(input_dims);
        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_leaky_relu((const float*)inputs[0], (float*)outputs[0],
                                 n_total, slope, batchSize, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_leaky_relu((const __half*)inputs[0], (__half*)outputs[0],
                                 n_total, slope, batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to leaky_relu::enqueue");
        }
        return 0;
    }
}
}
