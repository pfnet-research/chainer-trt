/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#define IS_TRT5RC (NV_TENSORRT_MAJOR == 5 &&           \
    (NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH < 2)) \

namespace chainer_trt {
namespace plugin {
    template <class T>
    class plugin_base : public nvinfer1::IPluginExt {
    protected:
        nvinfer1::PluginFormat plugin_format = nvinfer1::PluginFormat::kNCHW;
        nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;

    public:
        plugin_base() {}

        // Default: Plugin should support NCHW-ordered FP32/FP16
        bool supportsFormat(nvinfer1::DataType type,
                            nvinfer1::PluginFormat format) const override {
            if(format != nvinfer1::PluginFormat::kNCHW)
                return false;
            if(type != nvinfer1::DataType::kFLOAT &&
               type != nvinfer1::DataType::kHALF)
                return false;
            return true;
        }

        void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                 const nvinfer1::Dims* outputDims,
                                 int nbOutputs, nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format,
                                 int maxBatchSize) override {
            (void)inputDims;
            (void)nbInputs;
            (void)outputDims;
            (void)nbOutputs;
            (void)maxBatchSize;

            data_type = type;
            plugin_format = format;

            if(type != nvinfer1::DataType::kFLOAT &&
               type != nvinfer1::DataType::kHALF)
                throw std::runtime_error(
                  "Invalid DataType is specified to argmax_plugin::enqueue");
        }

        int getNbOutputs() const override { return 1; }

        int initialize() override { return 0; }

        void terminate() override {}

        size_t getWorkspaceSize(int) const override { return 0; }

        // Children has to override them instead of getPluginType and
        // getPluginVersion for better support for both TensorRT4 and 5
        virtual const char* get_plugin_type() const = 0;
        virtual const char* get_plugin_version() const = 0;

#if IS_TRT5RC
        const char* getPluginType() const override { return get_plugin_type(); }

        const char* getPluginVersion() const override {
            return get_plugin_version();
        }

        void destroy() override { delete this; }

        IPluginExt* clone() const override {
            // Clone object through serialization
            const size_t s = ((T*)this)->getSerializationSize();
            if(s == 0) {
                std::ostringstream ost;
                ost << "Error: serialization size of " << typeid(T).name();
                ost << " reported by getSerializationSize() is 0.";
                ost << " Something is wrong.";
                throw std::runtime_error(ost.str());
            }

            std::vector<unsigned char> buf(s, 0);
            // serialize
            ((T*)this)->serialize((void*)buf.data());

            // create new instance from the serialization
            return new T(buf.data(), s);
        }
#endif
    };
}
}
