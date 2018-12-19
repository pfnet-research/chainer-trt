/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <stdexcept>

#include <cuda_runtime.h>

#include <chainer_trt/chainer_trt.hpp>

#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/where.hpp"

namespace chainer_trt {
namespace plugin {

    where::where(nvinfer1::Dims _dims) : input_dims(_dims) {}

    where::where(const void* buf, size_t size) {
        (void)size;

        auto p = static_cast<const where*>(buf);
        input_dims = p->input_dims;
        data_type = p->data_type;
    }

    int where::enqueue(int batchSize, const void* const* inputs, void** outputs,
                       void* workspace, cudaStream_t stream) {
        (void)workspace;

        const int n_total = internal::calc_n_elements(input_dims);
        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_where((const float*)inputs[0], (const float*)inputs[1],
                            (const float*)inputs[2], (float*)outputs[0],
                            n_total, batchSize, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_where((const __half*)inputs[0], (const __half*)inputs[1],
                            (const __half*)inputs[2], (__half*)outputs[0],
                            n_total, batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to where::enqueue");
        }

        return 0;
    }

    void where::configureWithFormat(const nvinfer1::Dims* inputDims,
                                    int nbInputs,
                                    const nvinfer1::Dims* outputDims,
                                    int nbOutputs, nvinfer1::DataType type,
                                    nvinfer1::PluginFormat format,
                                    int maxBatchSize) {
        plugin_base::configureWithFormat(inputDims, nbInputs, outputDims,
                                         nbOutputs, type, format, maxBatchSize);
        this->input_dims = inputDims[0];
    }

    size_t where::getSerializationSize() { return sizeof(where); }

    void where::serialize(void* buffer) {
        auto p = static_cast<where*>(buffer);
        p->input_dims = input_dims;
        p->data_type = data_type;
    }
}
}
