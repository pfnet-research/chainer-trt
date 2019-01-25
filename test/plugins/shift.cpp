/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/shift.hpp"
#include "test_helper.hpp"

using chainer_trt::internal::make_dims;
using TestParams = std::tuple<int, std::string, nvinfer1::Dims, int, int, int,
                              int, nvinfer1::DataType>;

class ShiftKernelParameterizedTestTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(ShiftKernelParameterizedTestTest, CheckValues) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const auto dir = "test/fixtures/plugins/shift/" + std::get<1>(param);
    const nvinfer1::Dims dims = std::get<2>(param);
    const int kw = std::get<3>(param), kh = std::get<4>(param);
    const int dx = std::get<5>(param), dy = std::get<6>(param);
    const nvinfer1::DataType data_type = std::get<7>(param);

    chainer_trt::plugin::shift shift_src(dims, kw, kh, dx, dy);
    run_plugin_assert(shift_src, batch_size, data_type, dims, dir, 0.001);
}

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

INSTANTIATE_TEST_CASE_P(
  CheckOutputValue, ShiftKernelParameterizedTestTest,
  ::testing::Values(
    // FLOAT
    TestParams(1, "k3d1", make_dims(9, 10, 10), 3, 3, 1, 1, kFLOAT),
    TestParams(1, "k3d2", make_dims(9, 10, 10), 3, 3, 2, 2, kFLOAT),
    TestParams(1, "k3d3", make_dims(9, 10, 10), 3, 3, 3, 3, kFLOAT),
    TestParams(1, "k3d4", make_dims(9, 10, 10), 3, 3, 4, 4, kFLOAT),
    TestParams(1, "k5d1", make_dims(25, 10, 10), 5, 5, 1, 1, kFLOAT),
    TestParams(1, "k5d2", make_dims(25, 10, 10), 5, 5, 2, 2, kFLOAT),
    TestParams(1, "k5d3", make_dims(25, 10, 10), 5, 5, 3, 3, kFLOAT),
    TestParams(1, "kh3kw1d1", make_dims(9, 10, 10), 1, 3, 1, 1, kFLOAT),
    TestParams(1, "kh1kw3d1", make_dims(9, 10, 10), 3, 1, 1, 1, kFLOAT),
    TestParams(1, "k3dy3dx1", make_dims(9, 10, 10), 3, 3, 1, 3, kFLOAT),
    TestParams(1, "k3dy1dx3", make_dims(9, 10, 10), 3, 3, 3, 1, kFLOAT),

    // HALF
    TestParams(1, "k3d1", make_dims(9, 10, 10), 3, 3, 1, 1, kHALF),
    TestParams(1, "k3d2", make_dims(9, 10, 10), 3, 3, 2, 2, kHALF),
    TestParams(1, "k3d3", make_dims(9, 10, 10), 3, 3, 3, 3, kHALF),
    TestParams(1, "k3d4", make_dims(9, 10, 10), 3, 3, 4, 4, kHALF),
    TestParams(1, "k5d1", make_dims(25, 10, 10), 5, 5, 1, 1, kHALF),
    TestParams(1, "k5d2", make_dims(25, 10, 10), 5, 5, 2, 2, kHALF),
    TestParams(1, "k5d3", make_dims(25, 10, 10), 5, 5, 3, 3, kHALF),
    TestParams(1, "kh3kw1d1", make_dims(9, 10, 10), 1, 3, 1, 1, kHALF),
    TestParams(1, "kh1kw3d1", make_dims(9, 10, 10), 3, 1, 1, 1, kHALF),
    TestParams(1, "k3dy3dx1", make_dims(9, 10, 10), 3, 3, 1, 3, kHALF),
    TestParams(1, "k3dy1dx3", make_dims(9, 10, 10), 3, 3, 3, 1, kHALF),

    // FLOAT
    TestParams(32, "k3d1", make_dims(9, 10, 10), 3, 3, 1, 1, kFLOAT),
    TestParams(32, "k3d2", make_dims(9, 10, 10), 3, 3, 2, 2, kFLOAT),
    TestParams(32, "k3d3", make_dims(9, 10, 10), 3, 3, 3, 3, kFLOAT),
    TestParams(32, "k3d4", make_dims(9, 10, 10), 3, 3, 4, 4, kFLOAT),
    TestParams(32, "k5d1", make_dims(25, 10, 10), 5, 5, 1, 1, kFLOAT),
    TestParams(32, "k5d2", make_dims(25, 10, 10), 5, 5, 2, 2, kFLOAT),
    TestParams(32, "k5d3", make_dims(25, 10, 10), 5, 5, 3, 3, kFLOAT),
    TestParams(32, "kh3kw1d1", make_dims(9, 10, 10), 1, 3, 1, 1, kFLOAT),
    TestParams(32, "kh1kw3d1", make_dims(9, 10, 10), 3, 1, 1, 1, kFLOAT),
    TestParams(32, "k3dy3dx1", make_dims(9, 10, 10), 3, 3, 1, 3, kFLOAT),
    TestParams(32, "k3dy1dx3", make_dims(9, 10, 10), 3, 3, 3, 1, kFLOAT),

    // HALF
    TestParams(32, "k3d1", make_dims(9, 10, 10), 3, 3, 1, 1, kHALF),
    TestParams(32, "k3d2", make_dims(9, 10, 10), 3, 3, 2, 2, kHALF),
    TestParams(32, "k3d3", make_dims(9, 10, 10), 3, 3, 3, 3, kHALF),
    TestParams(32, "k3d4", make_dims(9, 10, 10), 3, 3, 4, 4, kHALF),
    TestParams(32, "k5d1", make_dims(25, 10, 10), 5, 5, 1, 1, kHALF),
    TestParams(32, "k5d2", make_dims(25, 10, 10), 5, 5, 2, 2, kHALF),
    TestParams(32, "k5d3", make_dims(25, 10, 10), 5, 5, 3, 3, kHALF),
    TestParams(32, "kh3kw1d1", make_dims(9, 10, 10), 1, 3, 1, 1, kHALF),
    TestParams(32, "kh1kw3d1", make_dims(9, 10, 10), 3, 1, 1, 1, kHALF),
    TestParams(32, "k3dy3dx1", make_dims(9, 10, 10), 3, 3, 1, 3, kHALF),
    TestParams(32, "k3dy1dx3", make_dims(9, 10, 10), 3, 3, 3, 1, kHALF)));
