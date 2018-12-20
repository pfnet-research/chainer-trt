/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

#include <chainer_trt/chainer_trt.hpp>
#include <chainer_trt/external/picojson_helper.hpp>

#include "include/chainer_trt_impl.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/constant_elementwise.hpp"

namespace chainer_trt {
namespace plugin {

    constant_elementwise::constant_elementwise(
      nvinfer1::Dims _dims, nvinfer1::ElementWiseOperation _op,
      const std::vector<float>& _values)
      : dims(_dims), op(_op), n_in(internal::calc_n_elements(dims)),
        values(_values) {
        if(values.size() != 1 && values.size() != n_in)
            throw std::invalid_argument(
              "Number of values in _values should be 1 or same as input "
              "dimension");
    }

    constant_elementwise::constant_elementwise(const void* buf, size_t size) {
        (void)size;

        const nvinfer1::Dims* p_dims = (const nvinfer1::Dims*)buf;
        dims = *p_dims;

        const nvinfer1::DataType* p_dt =
          (const nvinfer1::DataType*)(p_dims + 1);
        data_type = *p_dt;

        const nvinfer1::ElementWiseOperation* p_op =
          (const nvinfer1::ElementWiseOperation*)(p_dt + 1);
        op = *p_op;

        const int* p_n_values = (const int*)(p_op + 1);
        const int n_values = *p_n_values;

        const float* p_vals = (const float*)(p_n_values + 1);
        for(int i = 0; i < n_values; ++i)
            values.push_back(p_vals[i]);

        n_in = internal::calc_n_elements(dims);
    }

    nvinfer1::ILayer* constant_elementwise::build_layer(
      network_def network, const picojson::object& layer_params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names,
      const std::string& model_dir) {
        (void)dt;

        const auto type = param_get<std::string>(layer_params, "type");
        const auto source = param_get<std::string>(layer_params, "source");
        const auto constant_fn =
          param_get<std::string>(layer_params, "constant_weights_file");

        nvinfer1::ElementWiseOperation op;
        if(type == "AddConstant")
            op = nvinfer1::ElementWiseOperation::kSUM;
        else if(type == "SubFromConstant")
            op = nvinfer1::ElementWiseOperation::kSUB;
        else if(type == "MulConstant")
            op = nvinfer1::ElementWiseOperation::kPROD;
        else if(type == "DivFromConstant")
            op = nvinfer1::ElementWiseOperation::kDIV;
        else
            return nullptr;

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        // Load constant values
        internal::weights_manager weights;
        nvinfer1::Weights w = weights.load_weights_as(
          model_dir + "/" + constant_fn, nvinfer1::DataType::kFLOAT);
        std::vector<float> values;
        for(int i = 0; i < w.count; ++i)
            values.push_back(((float*)w.values)[i]);

        nvinfer1::ITensor* input = source_tensor->second;
        auto p =
          new plugin::constant_elementwise(input->getDimensions(), op, values);
        return network->addPlugin(&input, 1, *p);
    }

    int constant_elementwise::initialize() {
        // send values to GPU
        if(data_type == nvinfer1::DataType::kFLOAT) {
            cudaMalloc((void**)&values_gpu, sizeof(float) * values.size());
            cudaMemcpy(values_gpu, values.data(), sizeof(float) * values.size(),
                       cudaMemcpyHostToDevice);
        } else if(data_type == nvinfer1::DataType::kHALF) {
            const int n = values.size();
            std::vector<__half> values_half(n);
            internal::float2half(values.data(), values_half.data(), n);

            cudaMalloc((void**)&values_gpu, sizeof(__half) * values.size());
            cudaMemcpy(values_gpu, values_half.data(),
                       sizeof(__half) * values.size(), cudaMemcpyHostToDevice);
        } else {
            throw std::runtime_error(
              "Invalid DataType is specified to constant_elementwise");
        }
        return 0;
    }

    void constant_elementwise::terminate() {
        // release values in GPU
        cudaFree(values_gpu);
    }

    int constant_elementwise::enqueue(int batchSize, const void* const* inputs,
                                      void** outputs, void* workspace,
                                      cudaStream_t stream) {
        (void)workspace;

        if(data_type == nvinfer1::DataType::kFLOAT) {
            switch(op) {
                case nvinfer1::ElementWiseOperation::kSUM:
                    apply_eltw_sum((const float*)inputs[0], n_in, values_gpu,
                                   values.size(), (float*)outputs[0], batchSize,
                                   stream);
                    break;
                case nvinfer1::ElementWiseOperation::kSUB:
                    apply_eltw_sub((const float*)inputs[0], n_in, values_gpu,
                                   values.size(), (float*)outputs[0], batchSize,
                                   stream);
                    break;
                case nvinfer1::ElementWiseOperation::kPROD:
                    apply_eltw_mul((const float*)inputs[0], n_in, values_gpu,
                                   values.size(), (float*)outputs[0], batchSize,
                                   stream);
                    break;
                case nvinfer1::ElementWiseOperation::kDIV:
                    apply_eltw_div((const float*)inputs[0], n_in, values_gpu,
                                   values.size(), (float*)outputs[0], batchSize,
                                   stream);
                    break;
                default:
                    throw std::invalid_argument(
                      "Only Sum, Sub, Prod and Div are implemented");
            }
        } else if(data_type == nvinfer1::DataType::kHALF) {
            switch(op) {
                case nvinfer1::ElementWiseOperation::kSUM:
                    apply_eltw_sum((const __half*)inputs[0], n_in,
                                   (const __half*)values_gpu, values.size(),
                                   (__half*)outputs[0], batchSize, stream);
                    break;
                case nvinfer1::ElementWiseOperation::kSUB:
                    apply_eltw_sub((const __half*)inputs[0], n_in,
                                   (const __half*)values_gpu, values.size(),
                                   (__half*)outputs[0], batchSize, stream);
                    break;
                case nvinfer1::ElementWiseOperation::kPROD:
                    apply_eltw_mul((const __half*)inputs[0], n_in,
                                   (const __half*)values_gpu, values.size(),
                                   (__half*)outputs[0], batchSize, stream);
                    break;
                case nvinfer1::ElementWiseOperation::kDIV:
                    apply_eltw_div((const __half*)inputs[0], n_in,
                                   (const __half*)values_gpu, values.size(),
                                   (__half*)outputs[0], batchSize, stream);
                    break;
                default:
                    throw std::invalid_argument(
                      "Only Sum, Sub, Prod and Div are implemented");
            }
        } else {
            throw std::runtime_error(
              "Invalid DataType is specified to constant_elementwise::enqueue");
        }

        return 0;
    }

    size_t constant_elementwise::getSerializationSize() {
        return sizeof(dims) + sizeof(data_type) + sizeof(op) + sizeof(int) +
               sizeof(float) * values.size();
    }

    void constant_elementwise::serialize(void* buffer) {
        nvinfer1::Dims* p_dims = (nvinfer1::Dims*)buffer;
        *p_dims = dims;

        nvinfer1::DataType* p_dt = (nvinfer1::DataType*)(p_dims + 1);
        *p_dt = data_type;

        nvinfer1::ElementWiseOperation* p_op =
          (nvinfer1::ElementWiseOperation*)(p_dt + 1);
        *p_op = op;

        int* p_n_values = (int*)(p_op + 1);
        *p_n_values = values.size();

        float* p_vals = (float*)(p_n_values + 1);
        for(size_t i = 0; i < values.size(); ++i)
            p_vals[i] = values[i];
    }
}
}
