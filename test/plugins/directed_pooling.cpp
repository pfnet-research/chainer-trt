/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/directed_pooling.hpp"
#include "test_helper.hpp"

using chainer_trt::internal::make_dims;
using TestParams = std::tuple<int, std::string, nvinfer1::Dims, int, int,
                              nvinfer1::DataType, bool>;

class DirectedPoolingKernelParameterizedTestTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(DirectedPoolingKernelParameterizedTestTest, CheckValues) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const auto dir =
      "test/fixtures/plugins/directed_pooling/" + std::get<1>(param);
    const nvinfer1::Dims dims = std::get<2>(param);
    const int horizontal = std::get<3>(param);
    const int rev = std::get<4>(param);
    const nvinfer1::DataType data_type = std::get<5>(param);
    const bool load_binary = std::get<6>(param);

    chainer_trt::plugin::directed_pooling dp(dims, horizontal, rev);
    run_plugin_assert(dp, batch_size, data_type, dims, dir, 0.0, load_binary);
}

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

INSTANTIATE_TEST_CASE_P(
  CheckOutputValue, DirectedPoolingKernelParameterizedTestTest,
  ::testing::Values(
    // FLOAT
    TestParams(1, "left", make_dims(32, 20, 40), 1, 1, kFLOAT, true),
    TestParams(1, "right", make_dims(32, 20, 40), 1, 0, kFLOAT, true),
    TestParams(1, "top", make_dims(32, 20, 40), 0, 1, kFLOAT, true),
    TestParams(1, "bottom", make_dims(32, 20, 40), 0, 0, kFLOAT, true),
    // HALF
    TestParams(1, "left", make_dims(32, 20, 40), 1, 1, kHALF, true),
    TestParams(1, "right", make_dims(32, 20, 40), 1, 0, kHALF, true),
    TestParams(1, "top", make_dims(32, 20, 40), 0, 1, kHALF, true),
    TestParams(1, "bottom", make_dims(32, 20, 40), 0, 0, kHALF, true),

    TestParams(1, "left_small", make_dims(2, 3, 5), 1, 1, kFLOAT, false),
    TestParams(1, "right_small", make_dims(2, 3, 5), 1, 0, kFLOAT, false),
    TestParams(1, "top_small", make_dims(2, 3, 5), 0, 1, kFLOAT, false),
    TestParams(1, "bottom_small", make_dims(2, 3, 5), 0, 0, kFLOAT, false),

    TestParams(1, "left_small", make_dims(2, 3, 5), 1, 1, kHALF, false),
    TestParams(1, "right_small", make_dims(2, 3, 5), 1, 0, kHALF, false),
    TestParams(1, "top_small", make_dims(2, 3, 5), 0, 1, kHALF, false),
    TestParams(1, "bottom_small", make_dims(2, 3, 5), 0, 0, kHALF, false)));
