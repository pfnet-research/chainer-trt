/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_runtime.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/broadcast_to.hpp"

namespace chainer_trt {
namespace plugin {

    broadcast_to::broadcast_to(nvinfer1::Dims _in_dims,
                               nvinfer1::Dims _out_dims)
      : in_dims(_in_dims), out_dims(_out_dims) {}

    broadcast_to::broadcast_to(const void* buf, size_t size) {
        (void)size;

        auto p = static_cast<const broadcast_to*>(buf);
        in_dims = p->in_dims;
        out_dims = p->out_dims;
        data_type = p->data_type;
    }

    size_t broadcast_to::getSerializationSize() { return sizeof(broadcast_to); }

    nvinfer1::Dims
    broadcast_to::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                      int nbInputDims) {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        return this->out_dims;
    }

    int broadcast_to::initialize() {
        in_size = internal::calc_n_elements(in_dims);
        out_size = internal::calc_n_elements(out_dims);

        int i_nsize = in_size;
        int o_nsize = out_size;
        std::vector<int> i_d, o_d, i_strides, o_strides, i_axes;

        for(int i = 0; i < in_dims.nbDims; i++) {
            i_d.push_back(in_dims.d[i]);
            o_d.push_back(out_dims.d[i]);
            i_axes.push_back(i);
            i_nsize /= i_d[i];
            o_nsize /= o_d[i];
            i_strides.push_back(i_d[i] == 1 ? 0 : i_nsize);
            o_strides.push_back(o_nsize);
        }

        cudaMalloc((void**)&d_i_strides, i_strides.size() * sizeof(int));
        cudaMalloc((void**)&d_o_strides, o_strides.size() * sizeof(int));

        cudaMemcpy(d_i_strides, &i_strides[0], i_strides.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_o_strides, &o_strides[0], o_strides.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        return 0;
    }

    void broadcast_to::terminate() {
        cudaFree(d_i_strides);
        cudaFree(d_o_strides);
    }

    int broadcast_to::enqueue(int batchSize, const void* const* inputs,
                              void** outputs, void* workspace,
                              cudaStream_t stream) {
        (void)workspace;

        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_broadcast_to((const float*)inputs[0], (float*)outputs[0],
                                   d_i_strides, d_o_strides, in_size, out_size,
                                   in_dims.nbDims, batchSize, stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_broadcast_to((const __half*)inputs[0],
                                   (__half*)outputs[0], d_i_strides,
                                   d_o_strides, in_size, out_size,
                                   in_dims.nbDims, batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to broadcast_to::enqueue");
        }
        return 0;
    }

    void broadcast_to::configureWithFormat(
      const nvinfer1::Dims* inputDims, int nbInputs,
      const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type,
      nvinfer1::PluginFormat format, int maxBatchSize) {
        plugin_base::configureWithFormat(inputDims, nbInputs, outputDims,
                                         nbOutputs, type, format, maxBatchSize);

        // Ref:
        // https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
        // Two dimensions are compatible when
        // 1. they are equal, or
        // 2. one of them is 1   --> we only consider the case d_in == 1.
        for(int i = 0; i < inputDims[0].nbDims; i++) {
            const int d_in = inputDims[0].d[i];
            const int d_out = outputDims[0].d[i];
            (void)d_in;
            (void)d_out;
            assert(d_in == d_out or d_in == 1);
        }
    }

    void broadcast_to::serialize(void* buffer) {
        auto p = static_cast<broadcast_to*>(buffer);
        p->in_dims = this->in_dims;
        p->out_dims = this->out_dims;
        p->data_type = this->data_type;
    }
}
}
