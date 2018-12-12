/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>

#include "chainer_trt/chainer_trt.hpp"
#include "include/chainer_trt_impl.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "test_helper.hpp"

using chainer_trt::internal::make_dims;
using chainer_trt::plugin::slice;

using vecint = std::vector<int>;
using vecfloat = std::vector<float>;

class SliceTest : public ::testing::Test {};

// TODO: Support slices like [::-1], [:4:-1], [1::-1]
TEST_F(SliceTest, OpenSliceWithMinusStep) {
    slice s1(slice::optint(), slice::optint(), -1);
    EXPECT_THROW(s1.calculate_output_dim(10), std::runtime_error);

    // ar[:3:-1] -> [9, 8, 7, 6, 5, 4]
    slice s2(slice::optint(), 3, -1);
    EXPECT_THROW(s2.calculate_output_dim(10), std::runtime_error);

    // ar[3::-1] -> [3, 2, 1, 0]
    slice s3(3, slice::optint(), -1);
    EXPECT_THROW(s3.calculate_output_dim(10), std::runtime_error);
}

using SliceTestParams = std::tuple<slice, vecint>;

class SliceParameterizedTest
  : public ::testing::TestWithParam<SliceTestParams> {};

// This test is to confirm
// ar = list(arange(10))        # where 10 is the 2nd argument
// assert ar[(any slice)] == expected_src_indices_map
TEST_P(SliceParameterizedTest, OutputDimensionCalculationTest) {
    static const int input_dim = 10;
    auto param = GetParam();
    const slice s = std::get<0>(param);
    const std::vector<int> expected_src_indices_map = std::get<1>(param);

    EXPECT_EQ(s.calculate_output_dim(input_dim),
              expected_src_indices_map.size());
    std::vector<int> src_indices_map;
    s.foreach(input_dim,
              [&](int src_idx, int) { src_indices_map.push_back(src_idx); });
    assert_vector_eq(expected_src_indices_map, src_indices_map);
}

static auto none = slice::optint();
INSTANTIATE_TEST_CASE_P(
  OutputDimensionCalc, SliceParameterizedTest,
  ::testing::Values(
    SliceTestParams(slice(0, 10, 1), vecint{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
    SliceTestParams(slice(), vecint{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
    SliceTestParams(slice(0, 10, -1), vecint()),
    SliceTestParams(slice(9, 0, -1), vecint{9, 8, 7, 6, 5, 4, 3, 2, 1}),
    SliceTestParams(slice(0, 10, none), vecint{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
    SliceTestParams(slice(4, none, none), vecint{4, 5, 6, 7, 8, 9}),
    SliceTestParams(slice(none, 4, none), vecint{0, 1, 2, 3}),
    SliceTestParams(slice(none, none, 2), vecint{0, 2, 4, 6, 8}),
    SliceTestParams(slice(0, 10, 3), vecint{0, 3, 6, 9}),
    SliceTestParams(slice(9, 0, -3), vecint{9, 6, 3}),
    SliceTestParams(slice(9, 0, 1), vecint()),
    SliceTestParams(slice(0, -1, none), vecint{0, 1, 2, 3, 4, 5, 6, 7, 8}),
    SliceTestParams(slice(-1, 0, none), vecint()),
    SliceTestParams(slice(-1, -5, -1), vecint{9, 8, 7, 6}),

    // when the specified index is out of range
    SliceTestParams(slice(0, 15, 2), vecint{0, 2, 4, 6, 8}),
    SliceTestParams(slice(10, 15, 1), vecint()),
    SliceTestParams(slice(15, 10, -1), vecint()),
    SliceTestParams(slice(15, 5, -1), vecint{9, 8, 7, 6}),
    SliceTestParams(slice(-4, 15, 1), vecint{6, 7, 8, 9}),
    SliceTestParams(slice(-4, -15, -1), vecint{6, 5, 4, 3, 2, 1, 0})));

// integer indexing
INSTANTIATE_TEST_CASE_P(OutputDimensionCalcIntIndex, SliceParameterizedTest,
                        ::testing::Values(SliceTestParams(slice(0), vecint{0}),
                                          SliceTestParams(slice(5), vecint{5}),
                                          SliceTestParams(slice(-1),
                                                          vecint{9})));

class GetItemTest : public ::testing::Test {};

TEST_F(GetItemTest, SerializeDeserialize) {
    const std::vector<slice> slices = {
      // clang-format off
      slice(),
      slice(slice::optint(), 1, slice::optint()),
      slice(4, slice::optint(), slice::optint()),
      slice(slice::optint(), slice::optint(), 2)
      // clang-format on
    };
    const nvinfer1::Dims dims = make_dims(1, 3, 10, 10);

    chainer_trt::plugin::get_item src(dims, slices);
    std::unique_ptr<unsigned char[]> buf(
      new unsigned char[src.getSerializationSize()]);
    src.serialize(buf.get());

    chainer_trt::plugin::get_item dst(buf.get(), src.getSerializationSize());

    // Check if dimension is properly preserved
    assert_dims_eq(dst.get_input_dims(), make_dims(1, 3, 10, 10));

    // Check if slices are properly preserved
    ASSERT_EQ(dst.get_slices()[0], slices[0]);
    ASSERT_EQ(dst.get_slices()[1], slices[1]);
    ASSERT_EQ(dst.get_slices()[2], slices[2]);
    ASSERT_EQ(dst.get_slices()[3], slices[3]);

    // Check if mappings are properly preserved
    // numpy.arange(300).reshape((1, 3, 10, 10))[:, :1:, 4::, ::2].flatten()
    // -> [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    //     74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
    assert_vector_eq(src.generate_mappings(dims, slices),
                     dst.generate_mappings(dims, slices));
}

TEST_F(GetItemTest, GetDimensions) {
    // clang-format off
    const std::vector<slice> slices = {
      slice(),
      slice(slice::optint(), 1, slice::optint()),
      slice(4, slice::optint(), slice::optint()),
      slice(slice::optint(), slice::optint(), 2)
    };
    // clang-format on
    const nvinfer1::Dims dims = make_dims(1, 3, 10, 10);
    chainer_trt::plugin::get_item getitem(dims, slices);

    const nvinfer1::Dims out_dims = getitem.getOutputDimensions(0, &dims, 1);
    assert_dims_eq(out_dims, make_dims(1, 1, 6, 5));
}

using MappingTestParams =
  std::tuple<nvinfer1::Dims, std::vector<slice>, std::vector<int>>;

class GetItemGenerateMappingParameterizedTest
  : public ::testing::TestWithParam<MappingTestParams> {};

TEST_P(GetItemGenerateMappingParameterizedTest, MappingTest) {
    auto param = GetParam();
    const nvinfer1::Dims input_dims = std::get<0>(param);
    const std::vector<slice> slices = std::get<1>(param);
    const std::vector<int> expected_mapping = std::get<2>(param);

    chainer_trt::plugin::get_item getitem(input_dims, slices);
    const auto mapping = getitem.generate_mappings(input_dims, slices);
    assert_vector_eq(mapping, expected_mapping);
}

// equivalent to
// numpy.arange(300).reshape((1, 3, 10, 10))[(slices)].flatten()
INSTANTIATE_TEST_CASE_P(
  OutputDimensionCalc, GetItemGenerateMappingParameterizedTest,
  ::testing::Values(
    MappingTestParams(make_dims(1, 3, 10, 10),
                      // clang-format off
                      std::vector<slice>{
                          slice(),
                          slice(1, slice::optint(), slice::optint()),
                          slice(3, 5, slice::optint()),
                          slice(3, 6, 2)
                      },
                      // clang-format on
                      vecint{133, 135, 143, 145, 233, 235, 243, 245}),
    MappingTestParams(make_dims(1, 3, 10, 10),
                      // clang-format off
                      std::vector<slice>{
                          slice(),
                          slice(slice::optint(), 2, slice::optint()),
                          slice(-5, -3, slice::optint()),
                          slice(5, 3, -1)
                      },
                      // clang-format on
                      vecint{55, 54, 65, 64, 155, 154, 165, 164}),
    MappingTestParams(make_dims(1, 3, 10, 10),
                      // clang-format off
                      std::vector<slice>{
                          slice(0),
                          slice(-1),
                          slice(3, 5, slice::optint()),
                          slice(4, 8, slice::optint())
                      },
                      // clang-format on
                      vecint{234, 235, 236, 237, 244, 245, 246, 247})));

using TestParams =
  std::tuple<int, nvinfer1::DataType, nvinfer1::Dims, nvinfer1::Dims,
             std::vector<slice>, std::vector<float>>;

class GetItemApplySliceParameterizedTest
  : public ::testing::TestWithParam<TestParams> {};

static std::vector<float> arange(int n) {
    std::vector<float> dat(n);
    for(int i = 0; i < n; ++i)
        dat[i] = i;
    return dat;
}

// Test for slice result confirmation.
// Assuming that the input is arange ([0, 1, 2, 3, ...]),
// so it should be exactly same as mapping array
TEST_P(GetItemApplySliceParameterizedTest, ApplySlice) {
    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const nvinfer1::DataType data_type = std::get<1>(param);
    const nvinfer1::Dims input_dims = std::get<2>(param);
    const nvinfer1::Dims expected_output_dims = std::get<3>(param);
    const std::vector<slice> slices = std::get<4>(param);
    const auto expected_out_cpu = repeat_array(std::get<5>(param), batch_size);
    const int n_in = chainer_trt::internal::calc_n_elements(input_dims);

    // write input values and expected output
    char tmpl[] = "ctrtXXXXXX";
    const std::string tmpdir_name = mkdtemp(tmpl);
    write_vector_to_file(tmpdir_name + "/in.csv", arange(n_in));
    write_vector_to_file(tmpdir_name + "/out.csv", std::get<5>(param));

    chainer_trt::plugin::get_item getitem_src(input_dims, slices);
    run_plugin_assert(getitem_src, batch_size, data_type, input_dims,
                      tmpdir_name, 0.001);

    nvinfer1::Dims in_dims[] = {input_dims};
    assert_dims_eq(getitem_src.getOutputDimensions(0, in_dims, 1),
                   expected_output_dims);

    std::remove((tmpdir_name + "/in.csv").c_str());
    std::remove((tmpdir_name + "/out.csv").c_str());
    rmdir(tmpdir_name.c_str());
}

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

INSTANTIATE_TEST_CASE_P(
  ApplySlice, GetItemApplySliceParameterizedTest,
  ::testing::Values(
    // FP32
    // clang-format off
    TestParams(1, kFLOAT, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(1, slice::optint(), slice::optint()),
                   slice(3, 5, slice::optint()),
                   slice(3, 6, 2)
               },
               vecfloat{133, 135, 143, 145, 233, 235, 243, 245}),
    TestParams(1, kFLOAT, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(slice::optint(), 2, slice::optint()),
                   slice(-5, -3, slice::optint()),
                   slice(5, 3, -1)
               },
               vecfloat{55, 54, 65, 64, 155, 154, 165, 164}),
    TestParams(1, kFLOAT, make_dims(1, 3, 10, 10), make_dims(2, 4),
               std::vector<slice>{
                       slice(0),
                       slice(-1),
                       slice(3, 5, slice::optint()),
                       slice(4, 8, slice::optint())
               },
               vecfloat{234, 235, 236, 237, 244, 245, 246, 247}),
    TestParams(65535, kFLOAT, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(1, slice::optint(), slice::optint()),
                   slice(3, 5, slice::optint()),
                   slice(3, 6, 2)
               },
               vecfloat{133, 135, 143, 145, 233, 235, 243, 245}),
    TestParams(65535, kFLOAT, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(slice::optint(), 2, slice::optint()),
                   slice(-5, -3, slice::optint()),
                   slice(5, 3, -1)
               },
               vecfloat{55, 54, 65, 64, 155, 154, 165, 164}),
    TestParams(65535, kFLOAT, make_dims(1, 3, 10, 10), make_dims(2, 4),
               std::vector<slice>{
                       slice(0),
                       slice(-1),
                       slice(3, 5, slice::optint()),
                       slice(4, 8, slice::optint())
               },
               vecfloat{234, 235, 236, 237, 244, 245, 246, 247}),

    // HALF
    TestParams(1, kHALF, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(1, slice::optint(), slice::optint()),
                   slice(3, 5, slice::optint()),
                   slice(3, 6, 2)
               },
               vecfloat{133, 135, 143, 145, 233, 235, 243, 245}),
    TestParams(1, kHALF, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(slice::optint(), 2, slice::optint()),
                   slice(-5, -3, slice::optint()),
                   slice(5, 3, -1)
               },
               vecfloat{55, 54, 65, 64, 155, 154, 165, 164}),
    TestParams(1, kHALF, make_dims(1, 3, 10, 10), make_dims(2, 4),
               std::vector<slice>{
                       slice(0),
                       slice(-1),
                       slice(3, 5, slice::optint()),
                       slice(4, 8, slice::optint())
               },
               vecfloat{234, 235, 236, 237, 244, 245, 246, 247}),
    TestParams(65535, kHALF, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(1, slice::optint(), slice::optint()),
                   slice(3, 5, slice::optint()),
                   slice(3, 6, 2)
               },
               vecfloat{133, 135, 143, 145, 233, 235, 243, 245}),
    TestParams(65535, kHALF, make_dims(1, 3, 10, 10), make_dims(1, 2, 2, 2),
               std::vector<slice>{
                   slice(),
                   slice(slice::optint(), 2, slice::optint()),
                   slice(-5, -3, slice::optint()),
                   slice(5, 3, -1)
               },
               vecfloat{55, 54, 65, 64, 155, 154, 165, 164}),
    TestParams(65535, kHALF, make_dims(1, 3, 10, 10), make_dims(2, 4),
               std::vector<slice>{
                   slice(0),
                   slice(-1),
                   slice(3, 5, slice::optint()),
                   slice(4, 8, slice::optint())
               },
               vecfloat{234, 235, 236, 237, 244, 245, 246, 247})));
// clang-format on
