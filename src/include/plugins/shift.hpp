/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <chainer_trt/plugin.hpp>

namespace chainer_trt {
namespace plugin {
    class shift : public plugin_base<shift> {
        int kw, kh, dx, dy;
        int c, h, w;
        int block_size;
        int grid_size;

        void setLaunchConfiguration();

    public:
        shift(nvinfer1::Dims in_dim, int _kw, int _kh, int _dx, int _dy);
        shift(const void* buf, size_t size);

        static nvinfer1::ILayer*
        build_layer(network_def network, const picojson::object& layer_params,
                    nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                    const std::string& model_dir);

        int get_kw() const { return kw; }
        int get_kh() const { return kh; }
        int get_dx() const { return dx; }
        int get_dy() const { return dy; }
        int get_c() const { return c; }
        int get_h() const { return h; }
        int get_w() const { return w; }

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        size_t getSerializationSize() override { return sizeof(shift); }

        void serialize(void* buffer) override;

        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_shift";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
