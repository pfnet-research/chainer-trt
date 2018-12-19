/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chainer_trt/chainer_trt.hpp>
#include <chainer_trt/external/picojson_helper.hpp>

#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/slice.hpp"

namespace chainer_trt {
namespace plugin {

    bool slice::operator==(const slice& that) const {
        return start == that.start && end == that.end && step == that.step;
    }

    slice slice::normalize(int input_dim) const {
        if(is_int_index)
            return *this;

        int _start = (start ? *start : 0);
        int _end = (end ? *end : input_dim);
        int _step = (step ? *step : 1);

        if(_step < 0 && (!start || !end))
            throw std::runtime_error(
              "Currently slice operation with negative step has to be a "
              "complete form (eg. a[i:j:-1], instead of x[i::-1] or x[:j:-1]). "
              "Please consider using the equivalent complete form "
              "in Python side");

        // negative pattern
        if(_start < 0)
            _start = input_dim + _start;
        if(_end < 0)
            _end = input_dim + _end;
        return slice(_start, _end, _step);
    }

    int slice::calculate_output_dim(int input_dim) const {
        int n = 0;
        slice::foreach(input_dim, [&](int, int) { ++n; });
        return n;
    }

    get_item::get_item(nvinfer1::Dims _input_dims,
                       const std::vector<slice>& _slices)
      : input_dims(_input_dims), slices(_slices) {
        assert(input_dims.nbDims == static_cast<signed>(slices.size()));
    }

    get_item::get_item(const void* buf, size_t size) {
        (void)size;

        // deserialize dimension
        const nvinfer1::Dims* p_dims = (const nvinfer1::Dims*)buf;
        input_dims = *p_dims;

        const nvinfer1::DataType* p_dt =
          (const nvinfer1::DataType*)(p_dims + 1);
        data_type = *p_dt;

        // deserialize slices
        const slice* p_slices = (const slice*)(p_dt + 1);
        for(int i = 0; i < this->input_dims.nbDims; ++i, ++p_slices)
            slices.push_back(*p_slices);

        // Check if the entire data is properly read
        if((char*)buf + size != (char*)p_slices)
            throw std::runtime_error(
              "GetItem deserialization error: serialization size mismatch. "
              "Please try rebuilding the engine file");
    }

    nvinfer1::ILayer* get_item::build_layer(
      network_def network, const picojson::object& layer_params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names,
      const std::string& model_dir) {
        (void)dt;
        (void)model_dir;

        const auto source = param_get<std::string>(layer_params, "source");
        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        auto err_msg = "Each of GetItem parameter has to be "
                       "3-element list, or a single integer."
                       "Fix ModelRetriever";

        // Parse array of array to list of shape
        auto json_slices = param_get<picojson::array>(layer_params, "slices");
        std::vector<plugin::slice> slices;
        for(const picojson::value& json_s : json_slices) {
            if(json_s.is<picojson::array>()) {
                auto json_slice = json_s.get<picojson::array>();
                if(json_slice.size() != 3)
                    throw std::runtime_error(err_msg);

                auto getvalornull = [](const picojson::value& o) {
                    return o.is<picojson::null>() ? plugin::slice::optint()
                                                  : (int)o.get<double>();
                };
                plugin::slice s(getvalornull(json_slice[0]),
                                getvalornull(json_slice[1]),
                                getvalornull(json_slice[2]));
                slices.push_back(s);
            } else if(json_s.is<double>()) {
                slices.push_back(plugin::slice((int)json_s.get<double>()));
            } else {
                throw std::runtime_error(err_msg);
            }
        }

        nvinfer1::ITensor* input = source_tensor->second;
        auto p = new plugin::get_item(input->getDimensions(), slices);
        assert(input->getDimensions().nbDims ==
               static_cast<signed>(slices.size()));
        return network->addPlugin(&input, 1, *p);
    }

    int get_item::initialize() {
        auto mappings = generate_mappings(input_dims, slices);
        out_size = mappings.size();
        cudaMalloc((void**)&mappings_gpu, sizeof(int) * out_size);
        cudaMemcpy(mappings_gpu, mappings.data(), sizeof(int) * mappings.size(),
                   cudaMemcpyHostToDevice);
        return 0;
    }

    void get_item::terminate() {
        cudaFree(mappings_gpu);
        mappings_gpu = NULL;
    }

    int get_item::enqueue(int batchSize, const void* const* inputs,
                          void** outputs, void* workspace,
                          cudaStream_t stream) {
        (void)workspace;

        const int in_size = internal::calc_n_elements(input_dims);
        switch(data_type) {
            case nvinfer1::DataType::kFLOAT:
                apply_slice<float>((const float*)inputs[0], (float*)outputs[0],
                                   mappings_gpu, in_size, out_size, batchSize,
                                   stream);
                break;
            case nvinfer1::DataType::kHALF:
                apply_slice<__half>((const __half*)inputs[0],
                                    (__half*)outputs[0], mappings_gpu, in_size,
                                    out_size, batchSize, stream);
                break;
            default:
                throw std::runtime_error(
                  "Invalid DataType is specified to get_item::enqueue");
        }
        return 0;
    }

    size_t get_item::getSerializationSize() {
        return sizeof(input_dims) + sizeof(data_type) +
               sizeof(slice) * slices.size();
    }

    void get_item::serialize(void* buffer) {
        /* Save dims first, followed by slices, then mapping */

        // serialize input_dims
        nvinfer1::Dims* p_dims = (nvinfer1::Dims*)buffer;
        *p_dims = input_dims;

        nvinfer1::DataType* p_dt = (nvinfer1::DataType*)(p_dims + 1);
        *p_dt = data_type;

        // serialize slices
        slice* p_slices = (slice*)(p_dt + 1);
        for(size_t i = 0; i < slices.size(); ++i, ++p_slices)
            *p_slices = slices[i];
    }

    void gen_mapping_loop(const nvinfer1::Dims& src_dims,
                          const std::vector<slice>& slices,
                          std::vector<int>& mapping, int d = 0,
                          int src_offset = 0, int dst_offset = 0) {
        if(d == src_dims.nbDims) {
            // now src_offset and dst_offset indicate corresponding index of src
            // and dst array, respectively.
            // Actually dst_offset is not used, although it points correct
            // index.
            assert(static_cast<signed>(mapping.size()) == dst_offset);
            mapping.push_back(src_offset);
        } else {
            const slice s = slices[d].normalize(src_dims.d[d]);
            const int dst_dim_d = s.calculate_output_dim(src_dims.d[d]);

            // ref.index caluculation formula for coord p[] in
            // a multidimensional (shape=P[]) array:
            // idx = p[0]; for(int i=1; i<p.size(); ++i) idx = p[i] + P[i] *
            // idx;
            // https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
            s.foreach(src_dims.d[d], [=, &mapping](int src_idx, int dst_idx) {
                const int _src_offset =
                  (d == 0 ? src_idx : src_idx + src_dims.d[d] * src_offset);
                const int _dst_offset =
                  (d == 0 ? dst_idx : dst_idx + dst_dim_d * dst_offset);
                gen_mapping_loop(src_dims, slices, mapping, d + 1, _src_offset,
                                 _dst_offset);
            });
        }
    }

    std::vector<int>
    get_item::generate_mappings(const nvinfer1::Dims& src_dims,
                                const std::vector<slice>& _slices) const {
        std::vector<int> mapping;
        gen_mapping_loop(src_dims, _slices, mapping);
        return mapping;
    }

    void get_item::configureWithFormat(const nvinfer1::Dims* inputDims,
                                       int nbInputs,
                                       const nvinfer1::Dims* outputDims,
                                       int nbOutputs, nvinfer1::DataType type,
                                       nvinfer1::PluginFormat format,
                                       int maxBatchSize) {
        plugin_base::configureWithFormat(inputDims, nbInputs, outputDims,
                                         nbOutputs, type, format, maxBatchSize);

        assert(nbInputs == 1);
        assert(inputDims->nbDims == input_dims.nbDims);
        for(int i = 0; i < input_dims.nbDims; ++i)
            assert(inputDims->d[i] == input_dims.d[i]);
    }

    nvinfer1::Dims get_item::getOutputDimensions(int index,
                                                 const nvinfer1::Dims* inputs,
                                                 int nbInputDims) {
        (void)index;
        (void)nbInputDims;

        assert(nbInputDims == 1);
        nvinfer1::Dims _in_dims = inputs[0];
        nvinfer1::Dims out_dims;
        out_dims.nbDims = 0;

        for(int dim = 0; dim < _in_dims.nbDims; ++dim) {
            auto s = slices[dim];
            if(!s.is_int_index) {
                int d = s.calculate_output_dim(_in_dims.d[dim]);
                out_dims.d[out_dims.nbDims++] = d;
            }
            //_in_dims.d[dim] = slices[dim].calculate_output_dim(
            //        _in_dims.d[dim]);
        }

        return out_dims;
    }
}
}
