/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/argmax.hpp"
#include "test_helper.hpp"

using chainer_trt::internal::make_dims;
using TestParams =
  std::tuple<int, std::string, nvinfer1::Dims, nvinfer1::DataType>;

class ArgmaxPluginParameterizedTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(ArgmaxPluginParameterizedTest, CheckValues) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const auto dir = "test/fixtures/plugins/argmax/" + std::get<1>(param);
    const nvinfer1::Dims in_dims = std::get<2>(param);
    const nvinfer1::DataType data_type = std::get<3>(param);

    chainer_trt::plugin::argmax argmax_src(in_dims);
    run_plugin_assert(argmax_src, batch_size, data_type, in_dims, dir);
}

INSTANTIATE_TEST_CASE_P(
  CheckOutputValue, ArgmaxPluginParameterizedTest,
  ::testing::Values(
    // channel size is 1 -> should return a zero array
    TestParams(1, "1x5x5", make_dims(1, 5, 5), nvinfer1::DataType::kFLOAT),
    TestParams(1, "1x5x5", make_dims(1, 5, 5), nvinfer1::DataType::kHALF),

    TestParams(1, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kFLOAT),
    TestParams(2, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kFLOAT),
    TestParams(1, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kHALF),
    TestParams(2, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kHALF),

    TestParams(65535, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kFLOAT),
    TestParams(65535, "8x5x5", make_dims(8, 5, 5), nvinfer1::DataType::kHALF)));
