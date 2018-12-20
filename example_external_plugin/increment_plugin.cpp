/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <stdexcept>

#include <cuda_fp16.h>

#include <chainer_trt/chainer_trt.hpp>
#include <chainer_trt/external/picojson_helper.hpp>
#include <chainer_trt/plugin.hpp>

#include "cmdline.h"

using chainer_trt::plugin::name_tensor_map;
using chainer_trt::plugin::network_def;

// CUDA kernel declaration
template <typename T>
void increment(const T*, T*, int, cudaStream_t);

class increment_plugin
  : public chainer_trt::plugin::plugin_base<increment_plugin> {
    int input_size = 0;

public:
    increment_plugin() {}

    increment_plugin(const void* buf, size_t size) {
        // Write deserialization code here
        (void)buf;
        (void)size;
    }

    size_t getSerializationSize() override { return 0; }

    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                             const nvinfer1::Dims* outputDims, int nbOutputs,
                             nvinfer1::DataType type,
                             nvinfer1::PluginFormat format,
                             int maxBatchSize) override {
        (void)nbInputs;
        (void)nbOutputs;
        (void)outputDims;
        (void)type;
        (void)format;
        (void)maxBatchSize;

        input_size = chainer_trt::internal::calc_n_elements(inputDims[0]);
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                       int nbInputDims) override {
        (void)index;
        (void)inputs;
        (void)nbInputDims;

        return inputs[0];
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs,
                void* workspace, cudaStream_t stream) override {
        (void)batchSize;
        (void)workspace;

        increment((const float*)inputs[0], (float*)outputs[0], input_size,
                  stream);
        return 0;
    }

    void serialize(void* buffer) override {
        // Save (minimum) information needed to describe the operation
        (void)buffer;
    }
};

// Network builder function called from chainer-trt
// Can also be a static member of increment_plugin
// (like chainer-trt default plugins)
nvinfer1::ILayer* build_increment(network_def network,
                                  const picojson::object& layer_params,
                                  nvinfer1::DataType dt,
                                  const name_tensor_map& tensor_names,
                                  const std::string& model_dir) {
    (void)dt;
    (void)model_dir;

    auto source = chainer_trt::param_get<std::string>(layer_params, "source");
    auto source_tensor = tensor_names.find(source);
    if(source_tensor == tensor_names.end())
        return NULL;

    nvinfer1::ITensor* input = source_tensor->second;
    auto p = new increment_plugin();
    return network->addPluginExt(&input, 1, *p);
}

int main(int argc, char* argv[]) {
    gengetopt_args_info args;
    if(cmdline_parser(argc, argv, &args))
        return -1;

    auto factory = std::make_shared<chainer_trt::plugin::plugin_factory>();
    factory->add_builder_deserializer("Increment", build_increment,
                                      increment_plugin::deserialize);
    const double workspace_gb = 1.0;
    const int max_batch = 1;
    auto m = chainer_trt::model::build_fp32(args.dir_arg, workspace_gb,
                                            max_batch, factory);
    chainer_trt::infer rt(m);

    // initialize input by zero
    const std::vector<float> in_cpu(12, 0.0);

    // allocate output initialized by zero
    std::vector<float> out_cpu(in_cpu.size(), 0.0);

    // Run inference
    rt.infer_from_cpu(1, {{"input", in_cpu.data()}},
                      {{"output", out_cpu.data()}});

    // Print output (Confirm that output are incremented)
    for(unsigned i = 0; i < out_cpu.size(); ++i)
        std::cout << (i ? ", " : "") << out_cpu[i];
    std::cout << std::endl;
}