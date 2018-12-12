/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/chainer_trt_impl.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "test_helper.hpp"

// TODO: Merge to run_plugin_assert_core (need to let it accept multiple inputs)

template <typename FloatType, typename PluginType>
void run_plugin_assert_core_n(PluginType& plugin_src, int n, int batch_size,
                              const nvinfer1::Dims& in_dims,
                              const std::string& dir, float allowed_err = 0) {
    cudaSetDevice(0);

    std::vector<FloatType> in_cpu[n];
    for(int i = 0; i < n; i++)
        in_cpu[i] = repeat_array(
          load_values<FloatType>(dir + "/in" + std::to_string(i + 1) + ".csv"),
          batch_size);
    const std::vector<FloatType> expected_out_cpu =
      repeat_array(load_values<FloatType>(dir + "/out.csv"), batch_size);
    const int n_in = chainer_trt::internal::calc_n_elements(in_dims);

    // Make a plugin (and serialize, deserialize)
    std::unique_ptr<unsigned char[]> buf(
      new unsigned char[plugin_src.getSerializationSize()]);
    plugin_src.serialize(buf.get());
    PluginType plugin(buf.get(), plugin_src.getSerializationSize());
    plugin.initialize();

    const nvinfer1::Dims out_dims = plugin.getOutputDimensions(0, &in_dims, 1);
    const int n_out = chainer_trt::internal::calc_n_elements(out_dims);

    for(int i = 0; i < n; i++)
        ASSERT_EQ(in_cpu[i].size(), n_in * batch_size);
    ASSERT_EQ(expected_out_cpu.size(), n_out * batch_size);

    // Prepare data
    FloatType *in_gpu[n], *out_gpu;
    for(int i = 0; i < n; i++) {
        cudaMalloc((void**)&in_gpu[i], sizeof(FloatType) * in_cpu[i].size());
        cudaMemcpy(in_gpu[i], in_cpu[i].data(),
                   sizeof(FloatType) * in_cpu[i].size(),
                   cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&out_gpu, sizeof(FloatType) * expected_out_cpu.size());

    // Run inference
    void **ins_gpu = (void**)in_gpu, *outs_gpu[] = {out_gpu};
    plugin.enqueue(batch_size, ins_gpu, outs_gpu, NULL, 0);

    // Get result and assert
    std::vector<FloatType> out_cpu(expected_out_cpu.size());
    cudaMemcpy(out_cpu.data(), out_gpu, sizeof(FloatType) * out_cpu.size(),
               cudaMemcpyDeviceToHost);

    if(allowed_err < 1e-3)
        assert_vector_eq(out_cpu, expected_out_cpu);
    else
        assert_vector_near(out_cpu, expected_out_cpu, allowed_err);

    for(int i = 0; i < n; i++)
        cudaFree(in_gpu[i]);
    cudaFree(out_gpu);
}

template <typename PluginType>
void run_plugin_assert_n(PluginType& plugin_src, int n, int batch_size,
                         nvinfer1::DataType data_type,
                         const nvinfer1::Dims& in_dims, const std::string& dir,
                         float allowed_err = 0) {
    const auto out_dim = plugin_src.getOutputDimensions(0, &in_dims, 1);

    plugin_src.configureWithFormat(&in_dims, 1, &out_dim, 1, data_type,
                                   nvinfer1::PluginFormat::kNCHW, batch_size);

    if(data_type == nvinfer1::DataType::kFLOAT)
        run_plugin_assert_core_n<float>(plugin_src, n, batch_size, in_dims, dir,
                                        allowed_err);
    else if(data_type == nvinfer1::DataType::kHALF)
        run_plugin_assert_core_n<__half>(plugin_src, n, batch_size, in_dims,
                                         dir, allowed_err);
}

using TestParams = std::tuple<int, nvinfer1::DataType>;

class WhereKernelParameterizedTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(WhereKernelParameterizedTest, CheckValues) {
    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const nvinfer1::DataType data_type = std::get<1>(param);
    const nvinfer1::Dims dims = chainer_trt::internal::make_dims(2, 3, 4);
    const std::string dir = "test/fixtures/plugins/where";

    chainer_trt::plugin::where where_src(dims);
    run_plugin_assert_n(where_src, 3, batch_size, data_type, dims, dir, 0.001);
}

INSTANTIATE_TEST_CASE_P(
  CheckOutputValue, WhereKernelParameterizedTest,
  ::testing::Values(TestParams(1, nvinfer1::DataType::kFLOAT),
                    TestParams(1, nvinfer1::DataType::kHALF),

                    TestParams(65535, nvinfer1::DataType::kFLOAT),
                    TestParams(65535, nvinfer1::DataType::kHALF)));
