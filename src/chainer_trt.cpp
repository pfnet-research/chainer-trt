/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/external/picojson_helper.hpp"
#include "include/chainer_trt_impl.hpp"

namespace chainer_trt {
// Define the instance of logger singleton internal variable
std::shared_ptr<nvinfer1::ILogger> logger::logger_instance;

void chainer_trt::default_logger::log(Severity severity, const char* msg) {
    if(severity != Severity::kINFO) // suppress info-level messages
        std::cout << msg << std::endl;
}

std::shared_ptr<nvinfer1::ILogger>& logger::get_logger() {
    if(!logger_instance)
        logger_instance = std::make_shared<chainer_trt::default_logger>();
    return logger_instance;
}

void logger::set_logger(std::shared_ptr<nvinfer1::ILogger> p) {
    std::swap(p, logger_instance);
}

bool verbose = false;
void set_verbose(bool _verbose) {
    verbose = _verbose;
}

bool get_verbose() {
    return verbose;
}

layer_not_implemented::layer_not_implemented(const std::string& layer_name,
                                             const std::string& layer_type) {
    std::ostringstream oss;
    oss << "Layer type " << layer_type << " is not supported yet (";
    oss << layer_name << ")";
    err = oss.str();
}

// picojson_helper.hpp
template <>
int param_get<int>(const picojson::object& params, const std::string& key) {
    return (int)param_get<double>(params, key);
}
template <>
float param_get<float>(const picojson::object& params, const std::string& key) {
    return (float)param_get<double>(params, key);
}

namespace internal {
    nvinfer1::Dims shapes_to_dims(const picojson::array& shapes) {
        assert(shapes.size() <= nvinfer1::Dims::MAX_DIMS); // <=8

        nvinfer1::Dims dims;
        dims.nbDims = shapes.size();

        for(size_t j = 0; j < shapes.size(); j++) {
            dims.d[j] = shapes[j].get<double>();
            dims.type[j] = nvinfer1::DimensionType::kSPATIAL;
        }
        dims.type[0] = nvinfer1::DimensionType::kCHANNEL;
        return dims;
    }

    std::vector<std::string> split(const std::string& str, char sep) {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string buffer;
        while(std::getline(ss, buffer, sep))
            result.push_back(buffer);
        return result;
    }

    nvinfer1::Weights
    weights_manager::load_weights_as(const std::string& file_name,
                                     nvinfer1::DataType dtype) {
        if(chainer_trt::get_verbose())
            std::cerr << "loading weights from " << file_name << std::endl;

        std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
        if(!ifs)
            throw std::string("File couldn't be opened ") + file_name;

        // Load header
        int sizeof_value;
        ifs >> sizeof_value;

        std::string line;
        ifs >> line;
        int count = 1;
        for(const std::string s : split(line, ','))
            count *= std::stoi(s);

        // read one '\n' and throw it away
        const auto c = ifs.get();
        (void)c;
        assert(c == '\n');

        if(sizeof_value == 1 || dtype == nvinfer1::DataType::kINT8)
            throw std::runtime_error(
              "Loading from/as INT8 weights is not supported");
        else if(sizeof_value != 2 && sizeof_value != 4)
            throw std::runtime_error("unknown size of value in weights");
        else if(sizeof_value == 2 && dtype == nvinfer1::DataType::kFLOAT)
            throw std::invalid_argument(
              "dtype is kFLOAT although the weights has HALF values");

        // Load binary bytes
        void* buf = new char[sizeof_value * count];
        ifs.read((char*)buf, sizeof_value * count);

        // Convert FP32 weights to FP16 if necessary
        if(sizeof_value == 4 && dtype == nvinfer1::DataType::kHALF) {
            __half* buf2 = new __half[count];
            float2half((const float*)buf, buf2, count);
            delete[](char*) buf;
            buf = (void*)buf2;
        }

        nvinfer1::Weights w{dtype, buf, count};

        this->weights.push_back(w);

        return w;
    }

    weights_manager::~weights_manager() {
        for(nvinfer1::Weights w : weights)
            delete[](char*) w.values;
    }

    // This is used by buffer.cpp and infer.cpp as an internal error check util
    void throw_n_in_out_err(int n_given, int n_actual, std::string in_or_out) {
        std::stringstream oss;
        oss << "The number of " << in_or_out << " doesn't match (";
        oss << "model has " << n_actual << " " << in_or_out;
        oss << ", but specified " << n_given << ")" << std::endl;
        throw std::invalid_argument(oss.str());
    }

    int8_entropy_calibrator::int8_entropy_calibrator(
      const std::vector<nvinfer1::Dims>& _input_dims,
      std::shared_ptr<calibration_stream> _calib_stream,
      const std::string& out_cache_file)
      : calib_stream(_calib_stream), input_dims(_input_dims), idx(0) {
        if(!out_cache_file.empty()) {
            ofs.open(out_cache_file,
                     std::ofstream::binary | std::ofstream::trunc);
            if(!ofs.is_open())
                throw std::invalid_argument(
                  "Destination cache file cannot be opened");
        }

        for(const nvinfer1::Dims& dims : input_dims) {
            void* p;
            cudaMalloc(&p, calc_n_elements(dims) * sizeof(float));
            batch_bufs_gpu.push_back(p);
        }
    }

    int8_entropy_calibrator::~int8_entropy_calibrator() {
        for(void* p : batch_bufs_gpu)
            cudaFree(p);
    }

    bool int8_entropy_calibrator::getBatch(void* bindings[],
                                           const char* names[],
                                           int nbBindings) {
        (void)names;
        (void)nbBindings;

        assert(nbBindings == static_cast<signed>(batch_bufs_gpu.size()));
        if(idx == calib_stream->get_n_batch())
            return false;

        for(size_t i = 0; i < input_dims.size(); ++i) {
            auto dims = input_dims[i];
            const std::vector<int> dims_vec(dims.d, dims.d + dims.nbDims);

            std::vector<float> batch_buf(calc_n_elements(dims));
            calib_stream->get_batch(idx, i, dims_vec, (void*)batch_buf.data());
            cudaMemcpy(batch_bufs_gpu[i], (void*)batch_buf.data(),
                       batch_buf.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            bindings[i] = batch_bufs_gpu[i];
        }

        ++idx;
        return true;
    }

    void int8_entropy_calibrator::writeCalibrationCache(const void* cache,
                                                        size_t length) {
        if(!ofs.is_open())
            return;
        ofs.write(reinterpret_cast<const char*>(cache), length);
    }

    int8_entropy_calibrator_cached::int8_entropy_calibrator_cached(
      const std::string& in_cache_file) {
        ifs.open(in_cache_file, std::ofstream::binary);
        if(!ifs.is_open())
            throw std::invalid_argument("Cache file not found");
    }

    const void*
    int8_entropy_calibrator_cached::readCalibrationCache(size_t& length) {
        std::vector<char> dat;
        while(!ifs.eof())
            dat.push_back((char)ifs.get());

        length = dat.size() - 1;
        char* p = new char[length - 1];
        std::copy(dat.begin(), dat.end() - 1, p);
        return (void*)p;
    }

    int calc_n_elements(const nvinfer1::Dims& dims) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                               [](int t1, int t2) { return t1 * t2; });
    }

    int calc_n_elements(const std::vector<int>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               [](int t1, int t2) { return t1 * t2; });
    }
}
}
