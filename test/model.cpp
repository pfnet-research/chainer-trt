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
#include "chainer_trt/external/picojson.h"
#include "include/chainer_trt_impl.hpp"
#include "test_helper.hpp"

// int: batch size
// std::string base path to the fixture
// std::vector<std::string> input data CSV files
// std::vector<std::string> expected output value CSV files
// std::vector<nvinfer1::Dims> expected output dimensions
// nvinfer1::DataType In which mode will this test case run
// float tolerance of value assertion
// std::string calibration cache file (only used in INT8 mode)
// std::vector<std::string> external plugin name list
using TestParams =
  std::tuple<int, std::string, std::vector<std::string>,
             std::vector<std::string>, std::vector<nvinfer1::Dims>,
             nvinfer1::DataType, float, std::string, std::vector<std::string>>;

// tuple params:
// batch_size, model_path, input_file_list, expected_output_file,
// expected_output_dims, model_data_type, allowed_abs_error
class TensorRTBuilderTestFixture : public ::testing::TestWithParam<TestParams> {
public:
    std::shared_ptr<chainer_trt::model>
    make_model(const std::string& model_dir, nvinfer1::DataType dt,
               const std::string& int8_calib_cache, int batch_size,
               std::shared_ptr<chainer_trt::plugin::plugin_factory> factory) const {
        if(dt == nvinfer1::DataType::kFLOAT)
            return chainer_trt::model::build_fp32(
              model_dir, 2, batch_size, factory);
        else if(dt == nvinfer1::DataType::kHALF)
            return chainer_trt::model::build_fp16(
              model_dir, 2, batch_size, factory);
        else if(dt == nvinfer1::DataType::kINT8 && !int8_calib_cache.size())
            // If no calib cache is specified, use 1000 random data.
            return chainer_trt::model::build_int8(
              model_dir, std::make_shared<BS>(1000), 2, batch_size, "", factory);
        else if(dt == nvinfer1::DataType::kINT8 && int8_calib_cache.size())
            return chainer_trt::model::build_int8_cache(
              model_dir, int8_calib_cache, 2, batch_size, factory);
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
    const std::vector<std::string> expected_output_fns = std::get<3>(param);
    const std::vector<nvinfer1::Dims> expected_output_dims = std::get<4>(param);
    const nvinfer1::DataType model_mode = std::get<5>(param);
    const float allowed_relative_error = std::get<6>(param);
    const std::string int8_calib_cache = std::get<7>(param);
    const std::vector<std::string> external_plugin_names = std::get<8>(param);

    // Make plugin factory
    auto factory = std::make_shared<chainer_trt::plugin::plugin_factory>();
    for(auto plugin_name : external_plugin_names) {
        switch(plugin_name) {
          case "Increment3":
            // factory->add_builder_deserializer(plugin_name, , );
            break;
        }
    }

    const auto model_src = make_model(export_path, model_mode, int8_calib_cache,
                                      batch_size, factory);

    // Serialize
    std::ostringstream oss;
    model_src->serialize(oss);

    // Deserialize
    std::istringstream iss(oss.str());
    const auto model = chainer_trt::model::deserialize(iss, factory);

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

    // Get output dimension and check
    ASSERT_EQ(model->get_n_outputs(), expected_output_fns.size());
    ASSERT_EQ(model->get_n_outputs(), expected_output_dims.size());
    std::vector<std::vector<float>> expected_outputs, outputs;
    std::vector<void*> out_bufs;
    for(int out_idx = 0; out_idx < model->get_n_outputs(); ++out_idx) {
        int out_size = 1;
        const auto output_dim = model->get_output_dimensions(out_idx);
        ASSERT_EQ(output_dim.size(), expected_output_dims[out_idx].nbDims);
        for(unsigned i = 0; i < output_dim.size(); ++i) {
            ASSERT_EQ(output_dim[i], expected_output_dims[out_idx].d[i]);
            out_size *= output_dim[i];
        }
        auto expected_out_fn = expected_output_fns[out_idx];
        const auto out_vals = load_values<float>(export_path + expected_out_fn);
        const auto expected_output = repeat_array(out_vals, batch_size);
        expected_outputs.push_back(expected_output);

        outputs.push_back(std::vector<float>(out_size * batch_size, 0));
        out_bufs.push_back(outputs[out_idx].data());
    }

    // Run inference
    chainer_trt::infer rt(model);
    rt.infer_from_cpu(batch_size, input_bufs, out_bufs);

    // Assertion by comparing outputs
    for(int out_idx = 0; out_idx < model->get_n_outputs(); ++out_idx) {
        assert_values(expected_outputs[out_idx], outputs[out_idx],
                      allowed_relative_error);
    }
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

        std::vector<std::string> expected_outputs;
        auto outputs_obj =
          f.find("expected_outputs")->second.get<picojson::array>();
        for(auto out_obj : outputs_obj)
            expected_outputs.push_back(out_obj.get<std::string>());

        std::vector<nvinfer1::Dims> out_dims;
        for(auto dim : f.find("output_dims")->second.get<picojson::array>()) {
            auto dim_obj = dim.get<picojson::array>();
            nvinfer1::Dims out_dim;
            out_dim.nbDims = 0;
            for(auto outdim_obj : dim_obj)
                out_dim.d[out_dim.nbDims++] = int(outdim_obj.get<double>());
            out_dims.push_back(out_dim);
        }

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

        auto int8_calib_cache =
          f.find("int8_calib_cache")->second.get<std::string>();

        std::vector<std::string> external_plugins;
        for(auto p : f.find("external_plugins")->second.get<picojson::array>())
            external_plugins.push_back(p.get<std::string>());

        ret.push_back(TestParams((int)batch_size, name, inputs,
                                 expected_outputs, out_dims, dtype, error,
                                 int8_calib_cache, external_plugins));
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
