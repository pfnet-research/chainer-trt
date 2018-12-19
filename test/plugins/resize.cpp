/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/cuda/cuda_kernels.hpp"
#include "include/plugins/resize.hpp"
#include "test_helper.hpp"

using TestParams =
  std::tuple<int, std::string, int, int, int, int, int, nvinfer1::DataType>;

// param: path, n_channels, in_h, in_w, out_h, out_w
class ResizeKernelParameterizedTest
  : public ::testing::TestWithParam<TestParams> {};

TEST_P(ResizeKernelParameterizedTest, CheckValues) {
    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const auto dir = "test/fixtures/plugins/resize/" + std::get<1>(param);
    const int n_channels = std::get<2>(param);
    const int in_h = std::get<3>(param);
    const int in_w = std::get<4>(param);
    const int out_h = std::get<5>(param);
    const int out_w = std::get<6>(param);
    const nvinfer1::DataType data_type = std::get<7>(param);

    const nvinfer1::Dims in_dims =
      chainer_trt::internal::make_dims(n_channels, in_h, in_w);
    chainer_trt::plugin::resize resize_src(n_channels, in_h, in_w, out_h,
                                           out_w);
    // value range is 0-255 so absolute error of 0.3 is acceptable
    run_plugin_assert(resize_src, batch_size, data_type, in_dims, dir, 0.3);
}

static const auto kFLOAT = nvinfer1::DataType::kFLOAT;
static const auto kHALF = nvinfer1::DataType::kHALF;

/*
 * x1 = skimage.data.astronaut()
 * # original size 512x512 is too big for testing
 * x1 = skimage.transform.resize(x1, (32, 32), preserve_range=False)
 * x1 = Variable(x1.astype(np.float32).transpose(2, 0, 1)[None, :])
 * n, c, h, w = x1.shape
 * scale_h, scale_w = 2, 2      # <-- change scale
 * with chainer.using_config('train', False):
 *     with RetainHook():
 *         y = F.resize_images(x1, (int(h * scale_h), int(w * scale_w)))
 * retriever = ModelRetriever('relative/path/')(y).save()
 */
INSTANTIATE_TEST_CASE_P(
  ResizeCheckOutputValue, ResizeKernelParameterizedTest,
  ::testing::Values(
    // FLOAT
    TestParams(1, "identity_mapping", 3, 32, 32, 32, 32, kFLOAT),
    TestParams(1, "shrink_by_4", 3, 32, 32, 8, 8, kFLOAT),
    TestParams(1, "expand_by_2", 3, 32, 32, 64, 64, kFLOAT),
    TestParams(1, "expand_by_1_3", 3, 32, 32, 41, 41, kFLOAT), // w/ interp
    TestParams(1, "shrink_by_0_6", 3, 32, 32, 19, 19, kFLOAT), // w/ interp
    TestParams(1, "w1_3x_h0_6x", 3, 32, 32, 19, 41, kFLOAT),
    TestParams(1, "w0_6x_h1_3x", 3, 32, 32, 41, 19, kFLOAT),
    TestParams(1, "w2x_h1_3x", 3, 32, 32, 41, 64, kFLOAT),
    TestParams(1, "w1_3x_h2x", 3, 32, 32, 64, 41, kFLOAT),
    TestParams(1, "w0_6x_h0_3x", 3, 32, 32, 9, 19, kFLOAT),
    TestParams(1, "w0_3x_h0_6x", 3, 32, 32, 19, 9, kFLOAT),
    TestParams(1, "w1h1_2x", 3, 1, 1, 2, 2, kFLOAT),
    TestParams(1, "w2h1_2x", 3, 1, 2, 2, 4, kFLOAT),
    TestParams(1, "w1h2_2x", 3, 2, 1, 4, 2, kFLOAT),
    TestParams(8, "identity_mapping", 3, 32, 32, 32, 32, kFLOAT),
    TestParams(8, "shrink_by_4", 3, 32, 32, 8, 8, kFLOAT),
    TestParams(8, "expand_by_2", 3, 32, 32, 64, 64, kFLOAT),
    TestParams(8, "expand_by_1_3", 3, 32, 32, 41, 41, kFLOAT),
    TestParams(8, "shrink_by_0_6", 3, 32, 32, 19, 19, kFLOAT),
    TestParams(8, "w1_3x_h0_6x", 3, 32, 32, 19, 41, kFLOAT),
    TestParams(8, "w0_6x_h1_3x", 3, 32, 32, 41, 19, kFLOAT),
    TestParams(8, "w2x_h1_3x", 3, 32, 32, 41, 64, kFLOAT),
    TestParams(8, "w1_3x_h2x", 3, 32, 32, 64, 41, kFLOAT),
    TestParams(8, "w0_6x_h0_3x", 3, 32, 32, 9, 19, kFLOAT),
    TestParams(8, "w0_3x_h0_6x", 3, 32, 32, 19, 9, kFLOAT),
    TestParams(8, "w1h1_2x", 3, 1, 1, 2, 2, kFLOAT),
    TestParams(8, "w2h1_2x", 3, 1, 2, 2, 4, kFLOAT),
    TestParams(8, "w1h2_2x", 3, 2, 1, 4, 2, kFLOAT)));

INSTANTIATE_TEST_CASE_P(
  ResizeCheckOutputValueHALF, ResizeKernelParameterizedTest,
  ::testing::Values(
    // HALF
    TestParams(1, "identity_mapping", 3, 32, 32, 32, 32, kHALF),
    TestParams(1, "shrink_by_4", 3, 32, 32, 8, 8, kHALF),
    TestParams(1, "expand_by_2", 3, 32, 32, 64, 64, kHALF),
    TestParams(1, "expand_by_1_3", 3, 32, 32, 41, 41, kHALF), // w/ interp
    TestParams(1, "shrink_by_0_6", 3, 32, 32, 19, 19, kHALF), // w/ interp
    TestParams(1, "w1_3x_h0_6x", 3, 32, 32, 19, 41, kHALF),
    TestParams(1, "w0_6x_h1_3x", 3, 32, 32, 41, 19, kHALF),
    TestParams(1, "w2x_h1_3x", 3, 32, 32, 41, 64, kHALF),
    TestParams(1, "w1_3x_h2x", 3, 32, 32, 64, 41, kHALF),
    TestParams(1, "w0_6x_h0_3x", 3, 32, 32, 9, 19, kHALF),
    TestParams(1, "w0_3x_h0_6x", 3, 32, 32, 19, 9, kHALF),
    TestParams(1, "w1h1_2x", 3, 1, 1, 2, 2, kHALF),
    TestParams(1, "w2h1_2x", 3, 1, 2, 2, 4, kHALF),
    TestParams(1, "w1h2_2x", 3, 2, 1, 4, 2, kHALF),
    TestParams(8, "identity_mapping", 3, 32, 32, 32, 32, kHALF),
    TestParams(8, "shrink_by_4", 3, 32, 32, 8, 8, kHALF),
    TestParams(8, "expand_by_2", 3, 32, 32, 64, 64, kHALF),
    TestParams(8, "expand_by_1_3", 3, 32, 32, 41, 41, kHALF),
    TestParams(8, "shrink_by_0_6", 3, 32, 32, 19, 19, kHALF),
    TestParams(8, "w1_3x_h0_6x", 3, 32, 32, 19, 41, kHALF),
    TestParams(8, "w0_6x_h1_3x", 3, 32, 32, 41, 19, kHALF),
    TestParams(8, "w2x_h1_3x", 3, 32, 32, 41, 64, kHALF),
    TestParams(8, "w1_3x_h2x", 3, 32, 32, 64, 41, kHALF),
    TestParams(8, "w0_6x_h0_3x", 3, 32, 32, 9, 19, kHALF),
    TestParams(8, "w0_3x_h0_6x", 3, 32, 32, 19, 9, kHALF),
    TestParams(8, "w1h1_2x", 3, 1, 1, 2, 2, kHALF),
    TestParams(8, "w2h1_2x", 3, 1, 2, 2, 4, kHALF),
    TestParams(8, "w1h2_2x", 3, 2, 1, 4, 2, kHALF)));
