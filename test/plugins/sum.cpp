/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/sum.hpp"
#include "test_helper.hpp"

using chainer_trt::internal::make_dims;

using TestParams =
  std::tuple<int, std::string, nvinfer1::Dims, nvinfer1::DataType>;

class SumPluginParameterizedTest : public ::testing::TestWithParam<TestParams> {
};

TEST_P(SumPluginParameterizedTest, CheckValues) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const std::string dir = "test/fixtures/plugins/sum/" + std::get<1>(param);
    const nvinfer1::Dims in_dims = std::get<2>(param);
    const nvinfer1::DataType data_type = std::get<3>(param);

    chainer_trt::plugin::sum argmax_src(in_dims);
    run_plugin_assert(argmax_src, batch_size, data_type, in_dims, dir, 0.01);
}

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

INSTANTIATE_TEST_CASE_P(
  CheckOutputValue, SumPluginParameterizedTest,
  ::testing::Values(TestParams(1, "case1", make_dims(8, 12, 12), kFLOAT),
                    TestParams(1, "case1", make_dims(8, 12, 12), kHALF),

                    TestParams(1, "case2", make_dims(8, 2, 2, 2), kFLOAT),
                    TestParams(1, "case2", make_dims(8, 2, 2, 2), kHALF)));
