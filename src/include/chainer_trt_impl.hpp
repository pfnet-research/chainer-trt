/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <cstring>
#include <fstream>
#include <map>

#include <cuda_fp16.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

namespace chainer_trt {
namespace internal {
    nvinfer1::Dims shapes_to_dims(const picojson::array& shapes);

    class weights_manager {
        std::vector<nvinfer1::Weights> weights;

    public:
        nvinfer1::Weights load_weights_as(const std::string& file_name,
                                          nvinfer1::DataType dtype);
        ~weights_manager();
    };

    // This is used by buffer.cpp and infer.cpp as an internal error check util
    void throw_n_in_out_err(int n_given, int n_actual, std::string in_or_out);

    void float2half(const float* src, __half* dst, int n);
    void half2float(const __half* src, float* dst, int n);

    class output_dimensions_formula
      : public nvinfer1::IOutputDimensionsFormula {
        // See also:
        // https://github.com/chainer/chainer/blob/240a8021de9258fe1cec2e6d36c9124e3daba697/chainer/utils/conv.py
        int get_outsize(int size, int k, int s, int p, bool cover_all,
                        int d = 1) const {
            int dk = k + (k - 1) * (d - 1);
            if(cover_all)
                return (size + p * 2 - dk + s - 1) / s + 1;
            else
                return (size + p * 2 - dk) / s + 1;
        }

    public:
        std::map<std::string, bool> cover_all_flags;

        nvinfer1::DimsHW
        compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
                nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                nvinfer1::DimsHW dilation, const char* layerName) const {
            (void)dilation;

            auto cover_all = cover_all_flags.find(layerName);
            if(cover_all == cover_all_flags.end())
                throw std::runtime_error("Cover all flag was not found.");

            return nvinfer1::DimsHW{
              get_outsize(inputDims.h(), kernelSize.h(), stride.h(),
                          padding.h(), cover_all->second),
              get_outsize(inputDims.w(), kernelSize.w(), stride.w(),
                          padding.w(), cover_all->second)};
        }
    };

    struct build_context {
        // these are information needed to be kept until build process finishes
        std::shared_ptr<nvinfer1::IBuilder> builder;
        std::shared_ptr<nvinfer1::INetworkDefinition> network;
        std::shared_ptr<weights_manager> weights;
        std::shared_ptr<output_dimensions_formula> pool_outdim_formula;

        std::shared_ptr<model> build() {
            std::shared_ptr<model> rt_model(new model());
            rt_model->engine = unique_ptr_with_destroyer<nvinfer1::ICudaEngine>(
              builder->buildCudaEngine(*network));
            rt_model->set_n_inputs_and_outputs();
            return rt_model;
        }
    };

    // TODO: Support large batch size
    class int8_entropy_calibrator : public nvinfer1::IInt8EntropyCalibrator {
        std::shared_ptr<calibration_stream> calib_stream;
        std::vector<void*> batch_bufs_gpu;
        std::vector<nvinfer1::Dims> input_dims;
        int idx;
        std::ofstream ofs;

    public:
        int8_entropy_calibrator(
          const std::vector<nvinfer1::Dims>& _input_dims,
          std::shared_ptr<calibration_stream> _calib_stream,
          const std::string& out_cache_file);

        virtual ~int8_entropy_calibrator();

        int getBatchSize() const override {
            return calib_stream->get_batch_size();
        }

        bool getBatch(void* bindings[], const char* names[],
                      int nbBindings) override;

        const void* readCalibrationCache(size_t& length) override {
            (void)length;

            return NULL;
        }

        void writeCalibrationCache(const void* cache, size_t length) override;
    };

    class int8_entropy_calibrator_cached
      : public nvinfer1::IInt8EntropyCalibrator {
        std::ifstream ifs;

    public:
        int8_entropy_calibrator_cached(const std::string& in_cache_file);

        int getBatchSize() const override {
            throw std::runtime_error("This shouldn't be called");
        }

        bool getBatch(void* bindings[], const char* names[],
                      int nbBindings) override {
            (void)bindings;
            (void)names;
            (void)nbBindings;

            throw std::runtime_error("This shouldn't be called");
        }

        const void* readCalibrationCache(size_t& length) override;

        void writeCalibrationCache(const void* cache, size_t length) override {
            (void)cache;
            (void)length;

            // Do nothing even if it is called
        }
    };
}
}
