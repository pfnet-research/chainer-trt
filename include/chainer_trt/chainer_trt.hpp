/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#include <cuda_runtime_api.h>

#include "plugin.hpp"

namespace chainer_trt {
namespace internal {
    class build_context;

    // Custom deleter for TensorRT related objects
    template <typename T>
    struct tensorrt_object_destroyer {
        void operator()(T* t) const { t->destroy(); }
    };

    template <typename T>
    using unique_ptr_with_destroyer =
      std::unique_ptr<T, tensorrt_object_destroyer<T>>;

    template <int Depth, typename First>
    void _make_dims_internal(nvinfer1::Dims& dims, const First& first) {
        static_assert(Depth < 8,
                      "Number of arguments that can be "
                      "specified to make_dims is up to 8");
        static_assert(std::is_integral<First>::value,
                      "Arguments that can be specified to "
                      "make_dims must be integral type");

        dims.d[Depth] = first;
        dims.nbDims = Depth + 1;
        if(Depth == 0)
            dims.type[Depth] = nvinfer1::DimensionType::kCHANNEL;
        else
            dims.type[Depth] = nvinfer1::DimensionType::kSPATIAL;
    }

    template <int Depth, typename First, typename... Rest>
    void _make_dims_internal(nvinfer1::Dims& dims, const First& first,
                             const Rest&... rest) {
        _make_dims_internal<Depth>(dims, first);
        _make_dims_internal<Depth + 1>(dims, rest...);
    }

    // make_dims(1, 10, 10)
    // -> nvinfer1::Dims {
    //   nbDims=4,
    //   d=[1, 10, 10],
    //   type=[kINDEX, kCHANNEL, kSPATIAL]
    // }
    template <typename... Args>
    nvinfer1::Dims make_dims(const Args&... args) {
        nvinfer1::Dims dims;
        _make_dims_internal<0>(dims, args...);
        return dims;
    }

    int calc_n_elements(const nvinfer1::Dims& dims);
    int calc_n_elements(const std::vector<int>& dims);
}

class layer_not_implemented : public std::exception {
    std::string err;

public:
    layer_not_implemented(const std::string& layer_name,
                          const std::string& layer_type);

    const char* what() const noexcept { return err.c_str(); }
};

void set_verbose(bool _verbose);
bool get_verbose();

// public class
class calibration_stream {
public:
    virtual ~calibration_stream() = default;
    virtual int get_batch_size() { return 1; } // default
    virtual int get_n_batch() = 0;
    virtual int get_n_input() = 0;
    virtual void get_batch(int i_batch, int input_idx,
                           const std::vector<int>& dims, void* dst_buf_cpu) = 0;
};

class default_logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override;
};

class logger {
    static std::shared_ptr<nvinfer1::ILogger> logger_instance;

    logger() {}
    logger(const logger& other) = delete;
    logger& operator=(const logger& other) = delete;

public:
    static std::shared_ptr<nvinfer1::ILogger>& get_logger();
    static void set_logger(std::shared_ptr<nvinfer1::ILogger> p);
};

struct build_param {
    std::string model_dir;
    double workspace_gb;
    int max_batch_size;
    std::shared_ptr<plugin::plugin_factory> factory;

    build_param(const std::string& _model_dir, double ws = 6.0, int b = 1,
                std::shared_ptr<plugin::plugin_factory> _f =
                  std::make_shared<plugin::plugin_factory>())
      : model_dir(_model_dir), workspace_gb(ws), max_batch_size(b),
        factory(_f) {}
};

struct build_param_fp32 : build_param {
    using build_param::build_param; // ctor
};

struct build_param_fp16 : build_param {
    using build_param::build_param; // ctor
};

struct build_param_int8_cached : build_param {
    using build_param::build_param; // ctor

    std::string in_cache_file;

    build_param_int8_cached(const std::string& _m, const std::string& _in_cache)
      : build_param(_m), in_cache_file(_in_cache) {}
};

struct build_param_int8 : build_param {
    using build_param::build_param; // ctor

    std::shared_ptr<calibration_stream> calib_stream;
    std::string out_cache_file;

    build_param_int8(const std::string& _model_dir,
                     const std::string& _out_cache_file = "")
      : build_param(_model_dir), out_cache_file(_out_cache_file) {}
};

class infer;
class buffer;

class model {
    friend class internal::build_context;
    friend class infer;
    friend class buffer;

    internal::unique_ptr_with_destroyer<nvinfer1::ICudaEngine> engine;
    int n_inputs = 0;
    int n_outputs = 0;

    model();
    model(const model& that) = delete;
    model& operator=(const model& that) = delete;

    std::vector<int> get_binding_dimensions(int index) const;
    void set_n_inputs_and_outputs();

public:
    static std::shared_ptr<model> build(const build_param_fp32& param);
    static std::shared_ptr<model> build(const build_param_fp16& param);
    static std::shared_ptr<model> build(const build_param_int8& param);
    static std::shared_ptr<model> build(const build_param_int8_cached& param);

    static std::shared_ptr<model>
    build_fp32(const std::string& model_dir, double workspace_gb = 6.0,
               int max_batch_size = 1,
               std::shared_ptr<plugin::plugin_factory> factory =
                 std::make_shared<plugin::plugin_factory>());

    static std::shared_ptr<model>
    build_fp16(const std::string& model_dir, double workspace_gb = 6.0,
               int max_batch_size = 1,
               std::shared_ptr<plugin::plugin_factory> factory =
                 std::make_shared<plugin::plugin_factory>());

    static std::shared_ptr<model>
    build_int8(const std::string& model_dir,
               std::shared_ptr<calibration_stream> calib_stream,
               double workspace_gb = 6.0, int max_batch_size = 1,
               const std::string& out_cache_file = "",
               std::shared_ptr<plugin::plugin_factory> factory =
                 std::make_shared<plugin::plugin_factory>());

    static std::shared_ptr<model>
    build_int8_cache(const std::string& model_dir,
                     const std::string& in_cache_file,
                     double workspace_gb = 6.0, int max_batch_size = 1,
                     std::shared_ptr<plugin::plugin_factory> factory =
                       std::make_shared<plugin::plugin_factory>());

    static std::shared_ptr<model>
    deserialize(std::istream& ist,
                std::shared_ptr<plugin::plugin_factory> factory =
                  std::make_shared<plugin::plugin_factory>());
    static std::shared_ptr<model>
    deserialize(const std::string& model_file,
                std::shared_ptr<plugin::plugin_factory> factory =
                  std::make_shared<plugin::plugin_factory>());

    void serialize(std::ostream& ost) const;
    void serialize(const std::string& model_file) const;

    int get_max_batch_size() const;
    int get_n_inputs() const;
    int get_n_outputs() const;

    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;

    std::vector<int> get_input_dimensions(int input_index) const;
    std::vector<int> get_input_dimensions(const std::string& name) const;
    std::vector<int> get_output_dimensions(int output_index) const;
    std::vector<int> get_output_dimensions(const std::string& name) const;

    bool has_input(const std::string& input_name) const;
    bool has_output(const std::string& output_name) const;
};

class infer {
    using execution_context =
      internal::unique_ptr_with_destroyer<nvinfer1::IExecutionContext>;

    std::shared_ptr<model> m;
    std::shared_ptr<nvinfer1::IProfiler> profiler;
    execution_context cuda_context;
    infer() = delete;

public:
    bool debug = false;

    infer(std::shared_ptr<model> _m,
          std::shared_ptr<nvinfer1::IProfiler> _profiler = nullptr);

    // TODO: Implement
    infer(const infer& that) = delete;

    std::shared_ptr<buffer> create_buffer(int batch_size) const;

    void operator()(buffer& buf, cudaStream_t stream = 0);
    void operator()(int batch_size,
                    const std::map<std::string, const void*>& inputs_gpu,
                    const std::map<std::string, void*>& outputs_gpu,
                    cudaStream_t stream = 0);
    void operator()(int batch_size, const std::vector<const void*>& inputs_gpu,
                    const std::vector<void*>& outputs_gpu,
                    cudaStream_t stream = 0);
    void infer_from_cpu(int batch_size,
                        const std::vector<const void*>& inputs_cpu,
                        const std::vector<void*>& outputs_cpu,
                        cudaStream_t stream = 0);
    void infer_from_cpu(int batch_size,
                        const std::map<std::string, const void*>& inputs_gpu,
                        const std::map<std::string, void*>& outputs_gpu,
                        cudaStream_t stream = 0);

    /*
     * Interfaces to directly touch bindings. Not recommended.
     */
    std::vector<void*>
    create_bindings(const std::map<std::string, void*> name_buffer_map) const;

    std::vector<void*>
    create_bindings(const std::vector<void*>& inputs_gpu,
                    const std::vector<void*>& outputs_gpu) const;

    int get_binding_index(const std::string& name) const;

    void operator()(int batch_size, const std::vector<void*>& bindings,
                    cudaStream_t stream = 0);
};

class buffer {
    friend class infer;

    std::shared_ptr<model> m;
    int batch_size;
    std::vector<void*> gpu_buffers;
    std::vector<int> gpu_buffer_sizes;

    buffer() {}
    buffer(const buffer& that) = delete;
    buffer& operator=(const buffer&) = delete;

    int get_buffer_size(int buffer_idx, bool with_batch) const;

public:
    buffer(const buffer&& that);
    buffer(std::shared_ptr<model> _m, int _batch_size);

    int get_batch_size() const;

    void input_host_to_device(const std::vector<const void*>& inputs_cpu,
                              cudaStream_t stream = 0);
    void
    input_host_to_device(const std::map<std::string, const void*>& inputs_cpu,
                         cudaStream_t stream = 0);

    void output_device_to_host(const std::vector<void*>& outputs_cpu,
                               cudaStream_t stream = 0) const;
    void output_device_to_host(const std::map<std::string, void*>& outputs_cpu,
                               cudaStream_t stream = 0) const;

    int get_input_size(int input_idx, bool with_batch = true) const;
    int get_input_size(const std::string& input_name,
                       bool with_batch = true) const;
    int get_output_size(int output_idx, bool with_batch = true) const;
    int get_output_size(const std::string& output_name,
                        bool with_batch = true) const;
    void* get_input(int input_idx) const;
    void* get_input(const std::string& input_name) const;
    void* get_output(int output_idx) const;
    void* get_output(const std::string& output_name) const;

    ~buffer();
};
}
