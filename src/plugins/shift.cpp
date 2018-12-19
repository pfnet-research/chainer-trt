/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chainer_trt/external/picojson_helper.hpp>

#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/shift.hpp"

namespace chainer_trt {
namespace plugin {

    shift::shift(nvinfer1::Dims dims, int _kw, int _kh, int _dx, int _dy)
      : kw(_kw), kh(_kh), dx(_dx), dy(_dy), c(dims.d[0]), h(dims.d[1]),
        w(dims.d[2]) {
        this->setLaunchConfiguration();
    }

    nvinfer1::ILayer* shift::build_layer(network_def network,
                                         const picojson::object& layer_params,
                                         nvinfer1::DataType dt,
                                         const name_tensor_map& tensor_names,
                                         const std::string& model_dir) {
        (void)dt;
        (void)model_dir;

        const auto source = param_get<std::string>(layer_params, "source");
        const int kw = param_get<int>(layer_params, "kw");
        const int kh = param_get<int>(layer_params, "kh");
        const int dx = param_get<int>(layer_params, "dx");
        const int dy = param_get<int>(layer_params, "dy");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        auto p = new plugin::shift(input->getDimensions(), kw, kh, dx, dy);
        return network->addPluginExt(&input, 1, *p);
    }

    shift::shift(const void* buf, size_t size) {
        (void)size;
        assert(size == sizeof(shift));

        auto p = static_cast<const shift*>(buf);
        data_type = p->data_type;
        kw = p->kw;
        kh = p->kh;
        dx = p->dx;
        dy = p->dy;
        c = p->c;
        h = p->h;
        w = p->w;
        setLaunchConfiguration();
    }

    void shift::setLaunchConfiguration() {
        this->block_size = 512;
        int device;
        cudaGetDevice(&device);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
        this->grid_size = 8 * numSMs;
    }

    void shift::serialize(void* buffer) {
        auto p = static_cast<shift*>(buffer);
        p->data_type = data_type;
        p->kw = kw;
        p->kh = kh;
        p->dx = dx;
        p->dy = dy;
        p->c = c;
        p->h = h;
        p->w = w;
    }

    nvinfer1::Dims shift::getOutputDimensions(int index,
                                              const nvinfer1::Dims* inputs,
                                              int nbInputDims) {
        (void)index;
        (void)nbInputDims;

        return inputs[0];
    }

    int shift::enqueue(int batchSize, const void* const* inputs, void** outputs,
                       void* workspace, cudaStream_t stream) {
        (void)workspace;

        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_shift<float>((const float*)inputs[0], batchSize, c, h, w,
                                   kh, kw, dy, dx, grid_size, block_size,
                                   (float*)outputs[0], stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_shift<__half>((const __half*)inputs[0], batchSize, c, h,
                                    w, kh, kw, dy, dx, grid_size, block_size,
                                    (__half*)outputs[0], stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to shift::enqueue");
        }
        return 0;
    }
}
}
