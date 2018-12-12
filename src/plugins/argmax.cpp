/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_runtime.h>

#include "chainer_trt/chainer_trt.hpp"

#include "../include/cuda/cuda_kernels.hpp"
#include "../include/plugins/argmax.hpp"

namespace chainer_trt {
namespace plugin {

    argmax::argmax(nvinfer1::Dims _dims) : input_dims(_dims) {}

    argmax::argmax(const void *buf, size_t size) {
        (void)size;
        auto p = static_cast<const argmax *>(buf);
        input_dims = p->input_dims;
        data_type = p->data_type;
    }

    size_t argmax::getSerializationSize() { return sizeof(argmax); }

    nvinfer1::Dims argmax::getOutputDimensions(int index,
                                               const nvinfer1::Dims *inputs,
                                               int nbInputDims) {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        nvinfer1::Dims output_dims = input_dims;
        assert(output_dims.nbDims == 3); // Assuming CHW
        output_dims.d[0] = 1;
        return output_dims;
    }

    void argmax::serialize(void *buffer) {
        auto p = static_cast<argmax *>(buffer);
        p->input_dims = input_dims;
        p->data_type = data_type;
    }

    int argmax::enqueue(int batchSize, const void *const *inputs,
                        void **outputs, void *workspace, cudaStream_t stream) {
        (void)workspace;

        const int n_total = internal::calc_n_elements(input_dims);
        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_argmax((const float *)inputs[0], (float *)outputs[0],
                             input_dims.d[0], input_dims.d[1] * input_dims.d[2],
                             n_total, batchSize, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_argmax((const __half *)inputs[0], (__half *)outputs[0],
                             input_dims.d[0], input_dims.d[1] * input_dims.d[2],
                             n_total, batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to argmax::enqueue");
        }
        return 0;
    }
}
}
