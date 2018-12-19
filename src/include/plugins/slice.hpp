/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <experimental/optional>
#include <vector>

#include <chainer_trt/plugin.hpp>

namespace chainer_trt {
namespace plugin {
    struct slice {
        using optint = std::experimental::optional<int>;

        bool is_int_index = false;

        int int_index;

        optint start;
        optint end;
        optint step;

        slice() : is_int_index(false) {}
        slice(optint _start, optint _end, optint _step)
          : is_int_index(false), start(_start), end(_end), step(_step) {}
        slice(int i) : is_int_index(true), int_index(i) {}

        bool operator==(const slice& that) const;

        template <class F>
        void foreach(int input_dim, F f) const {
            if(is_int_index) {
                if(0 <= int_index)
                    f(int_index, 0);
                else
                    f(input_dim + int_index, 0);
            } else {
                const slice s = normalize(input_dim);
                const auto cond = (0 < *s.step)
                                    ? [](int i, int _end) { return i < _end; }
                                    : [](int i, int _end) { return _end < i; };
                for(int src_idx = *s.start, dst_idx = 0; cond(src_idx, *s.end);
                    src_idx += *s.step, ++dst_idx)
                    if(0 <= src_idx && src_idx < input_dim)
                        f(src_idx, dst_idx);
            }
        }

        slice normalize(int input_dim) const;
        int calculate_output_dim(int input_dim) const;
    };

    class get_item : public plugin_base<get_item> {
        nvinfer1::Dims input_dims;
        std::vector<slice> slices;
        int* mappings_gpu;
        int out_size; // this value is set by initialize()

    public:
        nvinfer1::Dims get_input_dims() const { return input_dims; }
        std::vector<slice> get_slices() const { return slices; }

        get_item(nvinfer1::Dims _input_dims, const std::vector<slice>& _slices);
        get_item(const void* buf, size_t size);

        int initialize() override;
        void terminate() override;

        int enqueue(int batchSize, const void* const* inputs, void** outputs,
                    void* workspace, cudaStream_t stream);

        size_t getSerializationSize() override;

        void serialize(void* buffer) override;

        std::vector<int>
        generate_mappings(const nvinfer1::Dims& src_dims,
                          const std::vector<slice>& _slices) const;

        void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                 const nvinfer1::Dims* outputDims,
                                 int nbOutputs, nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format,
                                 int maxBatchSize) override;

        nvinfer1::Dims getOutputDimensions(int index,
                                           const nvinfer1::Dims* inputs,
                                           int nbInputDims) override;

        const char* get_plugin_type() const override {
            return "chainer_trt_get_item";
        }

        const char* get_plugin_version() const override { return "1.0.0"; }
    };
}
}
