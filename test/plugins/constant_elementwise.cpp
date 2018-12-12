/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/chainer_trt_impl.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "test_helper.hpp"

using vec = std::vector<float>;
using TestParams = std::tuple<int, nvinfer1::DataType, vec, vec, vec,
                              nvinfer1::ElementWiseOperation>;

class ConstantElementWiseKernelTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(ConstantElementWiseKernelTest, CheckValues) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const nvinfer1::DataType data_type = std::get<1>(param);
    const auto in_cpu = repeat_array(std::get<2>(param), batch_size);
    const auto vals_cpu = std::get<3>(param);
    const auto expected_out_cpu = repeat_array(std::get<4>(param), batch_size);
    const nvinfer1::ElementWiseOperation op = std::get<5>(param);
    const auto in_dims =
      chainer_trt::internal::make_dims(std::get<2>(param).size(), 1, 1);

    // write input values and expected output
    char tmpl[] = "ctrtXXXXXX";
    const std::string tmpdir_name = mkdtemp(tmpl);
    write_vector_to_file(tmpdir_name + "/in.csv", std::get<2>(param));
    write_vector_to_file(tmpdir_name + "/out.csv", std::get<4>(param));

    chainer_trt::plugin::constant_elementwise constant_eltw_src(in_dims, op,
                                                                vals_cpu);
    run_plugin_assert(constant_eltw_src, batch_size, data_type, in_dims,
                      tmpdir_name, 0.01);

    std::remove((tmpdir_name + "/in.csv").c_str());
    std::remove((tmpdir_name + "/out.csv").c_str());
    rmdir(tmpdir_name.c_str());
}

static const auto kSUM = nvinfer1::ElementWiseOperation::kSUM;
static const auto kSUB = nvinfer1::ElementWiseOperation::kSUB;
static const auto kPROD = nvinfer1::ElementWiseOperation::kPROD;
static const auto kDIV = nvinfer1::ElementWiseOperation::kDIV;

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

static const vec vec1234{1, 2, 3, 4};
static const vec vec4444{4, 4, 4, 4};
static const vec vec4{4};

INSTANTIATE_TEST_CASE_P(
  ConstantArithmetic, ConstantElementWiseKernelTest,
  ::testing::Values(
    TestParams(1, kFLOAT, vec1234, vec4444, vec{5, 6, 7, 8}, kSUM),
    TestParams(1, kFLOAT, vec1234, vec{4}, vec{5, 6, 7, 8}, kSUM),
    TestParams(1, kFLOAT, vec1234, vec4444, vec{3, 2, 1, 0}, kSUB),
    TestParams(1, kFLOAT, vec1234, vec{4}, vec{3, 2, 1, 0}, kSUB),
    TestParams(1, kFLOAT, vec1234, vec4444, vec{4, 8, 12, 16}, kPROD),
    TestParams(1, kFLOAT, vec1234, vec{4}, vec{4, 8, 12, 16}, kPROD),
    TestParams(1, kFLOAT, vec1234, vec4444, vec{4, 2, 1.333, 1}, kDIV),
    TestParams(1, kFLOAT, vec1234, vec{4}, vec{4, 2, 1.333, 1}, kDIV),

    TestParams(1, kHALF, vec1234, vec4444, vec{5, 6, 7, 8}, kSUM),
    TestParams(1, kHALF, vec1234, vec{4}, vec{5, 6, 7, 8}, kSUM),
    TestParams(1, kHALF, vec1234, vec4444, vec{3, 2, 1, 0}, kSUB),
    TestParams(1, kHALF, vec1234, vec{4}, vec{3, 2, 1, 0}, kSUB),
    TestParams(1, kHALF, vec1234, vec4444, vec{4, 8, 12, 16}, kPROD),
    TestParams(1, kHALF, vec1234, vec{4}, vec{4, 8, 12, 16}, kPROD),
    TestParams(1, kHALF, vec1234, vec4444, vec{4, 2, 1.333, 1}, kDIV),
    TestParams(1, kHALF, vec1234, vec{4}, vec{4, 2, 1.333, 1}, kDIV),

    // max batch size 65535 (cuda limitation)
    TestParams(65535, kFLOAT, vec1234, vec4444, vec{5, 6, 7, 8}, kSUM),
    TestParams(65535, kFLOAT, vec1234, vec{4}, vec{5, 6, 7, 8}, kSUM),
    TestParams(65535, kFLOAT, vec1234, vec4444, vec{3, 2, 1, 0}, kSUB),
    TestParams(65535, kFLOAT, vec1234, vec{4}, vec{3, 2, 1, 0}, kSUB),
    TestParams(65535, kFLOAT, vec1234, vec4444, vec{4, 8, 12, 16}, kPROD),
    TestParams(65535, kFLOAT, vec1234, vec{4}, vec{4, 8, 12, 16}, kPROD),
    TestParams(65535, kFLOAT, vec1234, vec4444, vec{4, 2, 1.333, 1}, kDIV),
    TestParams(65535, kFLOAT, vec1234, vec{4}, vec{4, 2, 1.333, 1}, kDIV),

    TestParams(65535, kHALF, vec1234, vec4444, vec{5, 6, 7, 8}, kSUM),
    TestParams(65535, kHALF, vec1234, vec{4}, vec{5, 6, 7, 8}, kSUM),
    TestParams(65535, kHALF, vec1234, vec4444, vec{3, 2, 1, 0}, kSUB),
    TestParams(65535, kHALF, vec1234, vec{4}, vec{3, 2, 1, 0}, kSUB),
    TestParams(65535, kHALF, vec1234, vec4444, vec{4, 8, 12, 16}, kPROD),
    TestParams(65535, kHALF, vec1234, vec{4}, vec{4, 8, 12, 16}, kPROD),
    TestParams(65535, kHALF, vec1234, vec4444, vec{4, 2, 1.333, 1}, kDIV),
    TestParams(65535, kHALF, vec1234, vec{4}, vec{4, 2, 1.333, 1}, kDIV)));

class ConstantEltwPluginTest : public ::testing::Test {};

TEST_F(ConstantEltwPluginTest, ElementNumberCheck) {
    auto dims = chainer_trt::internal::make_dims(1, 2, 2);
    nvinfer1::ElementWiseOperation op = kSUM;

    // Empty values -> error
    ASSERT_ANY_THROW(
      { chainer_trt::plugin::constant_elementwise(dims, op, vec()); });

    // 2 values for 4 total input values -> error
    ASSERT_ANY_THROW({
        chainer_trt::plugin::constant_elementwise(dims, op, vec{0, 0});
    });

    // 1 values -> OK, 4 values -> OK
    ASSERT_NO_THROW(
      { chainer_trt::plugin::constant_elementwise(dims, op, vec{0}); });
    ASSERT_NO_THROW({
        chainer_trt::plugin::constant_elementwise(dims, op, vec{0, 1, 2, 3});
    });
}

TEST_F(ConstantEltwPluginTest, SerializeDeserialize) {
    auto dims = chainer_trt::internal::make_dims(1, 2, 2);
    nvinfer1::ElementWiseOperation op = kSUM;
    const vec vals = {1, 2, 3, 4};
    chainer_trt::plugin::constant_elementwise const_eltw_src(dims, op, vals);

    // serialize
    std::unique_ptr<unsigned char[]> buf(
      new unsigned char[const_eltw_src.getSerializationSize()]);
    const_eltw_src.serialize((void*)buf.get());

    // make another plugin instance by deserialize
    chainer_trt::plugin::constant_elementwise const_eltw(
      buf.get(), const_eltw_src.getSerializationSize());

    assert_dims_eq(dims, const_eltw.get_dims());
    ASSERT_EQ(op, const_eltw.get_op());
    assert_vector_eq(vals, const_eltw.get_values());
}
