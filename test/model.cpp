/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <cuda_runtime_api.h>

#include "chainer_trt/chainer_trt.hpp"
#include "include/chainer_trt_impl.hpp"
#include "include/picojson.h"
#include "test_helper.hpp"

using TestParams =
  std::tuple<int, std::string, std::vector<std::string>, std::string,
             nvinfer1::Dims, nvinfer1::DataType, float>;

// tuple params:
// batch_size, model_path, input_file_list, expected_output_file,
// expected_output_dims, model_data_type, allowed_abs_error
class TensorRTBuilderTestFixture : public ::testing::TestWithParam<TestParams> {
public:
    std::shared_ptr<chainer_trt::model> make_model(const std::string& model_dir,
                                                   nvinfer1::DataType dt,
                                                   int batch_size) const {
        if(dt == nvinfer1::DataType::kFLOAT)
            return chainer_trt::model::build_fp32(model_dir, 2, batch_size);
        else if(dt == nvinfer1::DataType::kHALF)
            return chainer_trt::model::build_fp16(model_dir, 2, batch_size);
        else if(dt == nvinfer1::DataType::kINT8)
            return chainer_trt::model::build_int8(
              model_dir, std::make_shared<BS>(1000), 2, batch_size);
        return std::shared_ptr<chainer_trt::model>();
    }

    void assert_values(const std::vector<float>& expected_output,
                       const std::vector<float>& output,
                       float allowed_error) const {
        if(allowed_error == 0)
            assert_vector_eq(expected_output, output);
        else
            assert_vector_near(expected_output, output, allowed_error);
    }
};

TEST_P(TensorRTBuilderTestFixture, TestWithSerialize) {
    cudaSetDevice(0);

    auto param = GetParam();
    const int batch_size = std::get<0>(param);
    const std::string base_path = "test/fixtures/model/";
    const std::string export_path = base_path + std::get<1>(param) + "/";
    const std::vector<std::string> inputs = std::get<2>(param);
    const std::string expected_output_fn = std::get<3>(param);
    const nvinfer1::Dims expected_output_dims = std::get<4>(param);
    const nvinfer1::DataType model_mode = std::get<5>(param);
    const float allowed_relative_error = std::get<6>(param);

    const auto model_src = make_model(export_path, model_mode, batch_size);

    // Serialize
    std::ostringstream oss;
    model_src->serialize(oss);

    // Deserialize
    std::istringstream iss(oss.str());
    const auto model = chainer_trt::model::deserialize(iss);
    chainer_trt::infer rt(model);

    // Load expected output
    const auto out_vals = load_values<float>(export_path + expected_output_fn);
    const auto expected_output = repeat_array(out_vals, batch_size);

    // Get output dimension and check
    int n_output = 1;
    const auto output_dim = model->get_output_dimensions(0);
    ASSERT_EQ(output_dim.size(), expected_output_dims.nbDims);
    for(unsigned i = 0; i < output_dim.size(); ++i) {
        ASSERT_EQ(output_dim[i], expected_output_dims.d[i]);
        n_output *= output_dim[i];
    }
    ASSERT_EQ(n_output * batch_size, expected_output.size());

    // Load inputs and check dimensions
    ASSERT_EQ(model->get_n_inputs(), inputs.size());
    std::vector<std::vector<float>> input_vecs;
    std::vector<const void*> input_bufs;
    for(unsigned i = 0; i < inputs.size(); ++i) {
        const std::vector<int> dims = model->get_input_dimensions(i);
        const int n_input_i = chainer_trt::internal::calc_n_elements(dims);

        const auto in_vals = load_values<float>(export_path + inputs[i]);
        input_vecs.push_back(repeat_array(in_vals, batch_size));
        input_bufs.push_back(input_vecs.back().data());
        ASSERT_EQ(input_vecs.back().size(), n_input_i * batch_size);
    }

    // Run inference
    std::vector<float> output(n_output * batch_size);
    std::vector<void*> out_dst{output.data()};
    rt.infer_from_cpu(batch_size, input_bufs, out_dst);

    assert_values(expected_output, output, allowed_relative_error);
}

// Load parameterized test cases from JSON file
std::vector<TestParams> load_params(const std::string& fixture_json_file) {
    std::vector<TestParams> ret;

    std::ifstream fs(fixture_json_file);
    if(!fs)
        throw std::invalid_argument(fixture_json_file + " cannot be opened");

    picojson::value json;
    fs >> json;
    fs.close();

    for(auto fixture : json.get<picojson::array>()) {
        auto f = fixture.get<picojson::object>();
        auto name = f.find("fixture")->second.get<std::string>();

        std::vector<std::string> inputs;
        auto inputs_obj = f.find("inputs")->second.get<picojson::array>();
        for(auto in_obj : inputs_obj)
            inputs.push_back(in_obj.get<std::string>());

        auto expected_output =
          f.find("expected_output")->second.get<std::string>();

        nvinfer1::Dims out_dim;
        out_dim.nbDims = 0;
        auto outdims_obj = f.find("output_dims")->second.get<picojson::array>();
        for(auto outdim_obj : outdims_obj)
            out_dim.d[out_dim.nbDims++] = int(outdim_obj.get<double>());

        auto batch_size = int(f.find("batch_size")->second.get<double>());
        auto error = f.find("acceptable_absolute_error")->second.get<double>();

        auto dtype_str = f.find("dtype")->second.get<std::string>();
        nvinfer1::DataType dtype;
        if(dtype_str == "kFLOAT")
            dtype = nvinfer1::DataType::kFLOAT;
        else if(dtype_str == "kHALF")
            dtype = nvinfer1::DataType::kHALF;
        else if(dtype_str == "kINT8")
            dtype = nvinfer1::DataType::kINT8;
        else
            throw std::runtime_error(
              "Only \"kFLOAT\", \"kHALF\" and \"kINT8\""
              " are supported for now");

        ret.push_back(TestParams((int)batch_size, name, inputs, expected_output,
                                 out_dim, dtype, error));
    }

    return ret;
}

// A hack to let chainer-trt recognize test case name
// (needed for running certain test cases by filtering option)
// https://stackoverflow.com/questions/46023379
struct PrintToStringParamName {
    template <typename T>
    std::string operator()(const ::testing::TestParamInfo<T>& info) const {
        auto bs = std::get<0>(info.param);
        auto name = std::get<1>(info.param);
        auto dt = std::get<5>(info.param);

        std::ostringstream oss;
        oss << name;
        oss << "_bs" << bs;
        switch(dt) {
            case nvinfer1::DataType::kFLOAT:
                oss << "_kFLOAT";
                break;
            case nvinfer1::DataType::kHALF:
                oss << "_kHALF";
                break;
            case nvinfer1::DataType::kINT8:
                oss << "_kINT8";
                break;
            default:
                oss << "_UnknownDataType";
        }
        return oss.str();
    }
};

/*
 * Run all the test cases!!!!!!!
 * This assumes test fixtures are already generated by test/make_test_cases.py
 * Be noted that those fixtures will be copied to build directory by cmake,
 * so every time you defined a new test cases in make_test_cases.py,
 * you have to re-run test/make_test_cases.py
 */
INSTANTIATE_TEST_CASE_P(
  ModelValueCheck, TensorRTBuilderTestFixture,
  ::testing::ValuesIn(load_params("test/fixtures/model_fixtures.json")),
  PrintToStringParamName());

// TODO: Deal with death tests
