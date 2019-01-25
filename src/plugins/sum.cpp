/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_runtime.h>

#include <chainer_trt/chainer_trt.hpp>
#include <chainer_trt/external/picojson_helper.hpp>

#include "include/chainer_trt_impl.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/sum.hpp"

namespace chainer_trt {
namespace plugin {

    sum::sum(nvinfer1::Dims _dims) : input_dims(_dims) {}

    sum::sum(const void* buf, size_t size) {
        (void)size;
        auto p = static_cast<const sum*>(buf);
        input_dims = p->input_dims;
        data_type = p->data_type;
    }

    nvinfer1::ILayer* sum::build_layer(network_def network,
                                       const picojson::object& layer_params,
                                       nvinfer1::DataType dt,
                                       const name_tensor_map& tensor_names,
                                       const std::string& model_dir) {
        (void)dt;
        (void)model_dir;

        const auto source = param_get<std::string>(layer_params, "source");
        const auto shapes = param_get<picojson::array>(layer_params, "shape");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        auto p = new plugin::sum(internal::shapes_to_dims(shapes));
        return network->addPluginExt(&input, 1, *p);
    }

    size_t sum::getSerializationSize() { return sizeof(sum); }

    nvinfer1::Dims sum::getOutputDimensions(int index,
                                            const nvinfer1::Dims* inputs,
                                            int nbInputDims) {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        nvinfer1::Dims output_dims;
        output_dims.nbDims = input_dims.nbDims - 1;
        for(int i = 1; i < input_dims.nbDims; i++) {
            output_dims.d[i - 1] = input_dims.d[i];
            output_dims.type[i - 1] = input_dims.type[i];
        }

        return output_dims;
    }

    void sum::serialize(void* buffer) {
        auto p = static_cast<sum*>(buffer);
        p->input_dims = input_dims;
        p->data_type = data_type;
    }

    int sum::enqueue(int batchSize, const void* const* inputs, void** outputs,
                     void* workspace, cudaStream_t stream) {
        (void)workspace;

        const int n_total = internal::calc_n_elements(input_dims);
        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_sum((const float*)inputs[0], (float*)outputs[0],
                          input_dims.d[0], n_total / input_dims.d[0], n_total,
                          batchSize, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_sum((const __half*)inputs[0], (__half*)outputs[0],
                          input_dims.d[0], n_total / input_dims.d[0], n_total,
                          batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to sum::enqueue");
        }
        return 0;
    }
}
}
