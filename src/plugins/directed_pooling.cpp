/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chainer_trt/external/picojson_helper.hpp>

#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/directed_pooling.hpp"

namespace chainer_trt {
namespace plugin {

    directed_pooling::directed_pooling(nvinfer1::Dims in_dim, int horizontal_,
                                       int rev_)
      : C(in_dim.d[0]), H(in_dim.d[1]), W(in_dim.d[2]), horizontal(horizontal_),
        rev(rev_) {}

    directed_pooling::directed_pooling(const void *buf, size_t size) {
        (void)size;
        assert(size == sizeof(directed_pooling));

        auto p = static_cast<const directed_pooling *>(buf);
        data_type = p->data_type;
        C = p->C;
        H = p->H;
        W = p->W;
        horizontal = p->horizontal;
        rev = p->rev;
    }

    nvinfer1::ILayer* directed_pooling::build_layer(
      network_def network, const picojson::object& layer_params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names,
      const std::string& model_dir) {
        (void)dt;
        (void)model_dir;

        const auto source = param_get<std::string>(layer_params, "source");

        // left, right, top, bottom
        const auto dir = param_get<std::string>(layer_params, "dir");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        int horizontal = 0;
        int rev = 0;
        if(dir == "left") {
            horizontal = 1;
            rev = 1;
        } else if(dir == "right") {
            horizontal = 1;
            rev = 0;
        } else if(dir == "top") {
            horizontal = 0;
            rev = 1;
        } else if(dir == "bottom") {
            horizontal = 0;
            rev = 0;
        } else {
            throw std::runtime_error(
              "Unknown dir parameter for directed_pooling.");
        }

        nvinfer1::ITensor* input = source_tensor->second;
        auto p =
          new plugin::directed_pooling(input->getDimensions(), horizontal, rev);
        return network->addPluginExt(&input, 1, *p);
    }

    void directed_pooling::serialize(void *buffer) {
        auto p = static_cast<directed_pooling *>(buffer);
        p->data_type = data_type;
        p->C = C;
        p->H = H;
        p->W = W;
        p->horizontal = horizontal;
        p->rev = rev;
    }

    nvinfer1::Dims directed_pooling::getOutputDimensions(
      int index, const nvinfer1::Dims *inputs, int nbInputDims) {
        (void)index;
        (void)nbInputDims;

        return inputs[0];
    }

    int directed_pooling::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
        (void)workspace;

        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_directed_pooling<float>(
                  (const float *)inputs[0], batchSize, C, H, W,
                  (bool)horizontal, (bool)rev, (float *)outputs[0], stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_directed_pooling<__half>(
                  (const __half *)inputs[0], batchSize, C, H, W,
                  (bool)horizontal, (bool)rev, (__half *)outputs[0], stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to directed_pooling::enqueue");
        }
        return 0;
    }
}
}
