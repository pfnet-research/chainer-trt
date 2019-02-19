/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <gtest/gtest.h>

#include "chainer_trt/chainer_trt.hpp"
#include "test_helper.hpp"

class ChainerTRT_HPP_Test : public ::testing::Test {};

TEST_F(ChainerTRT_HPP_Test, MakeDims) {
    auto t = chainer_trt::internal::make_dims(10, 20, 30, 40);
    ASSERT_EQ(t.nbDims, 4);
    ASSERT_EQ(t.d[0], 10);
    ASSERT_EQ(t.type[0], nvinfer1::DimensionType::kCHANNEL);
    ASSERT_EQ(t.d[1], 20);
    ASSERT_EQ(t.type[1], nvinfer1::DimensionType::kSPATIAL);
    ASSERT_EQ(t.d[2], 30);
    ASSERT_EQ(t.type[2], nvinfer1::DimensionType::kSPATIAL);
    ASSERT_EQ(t.d[3], 40);
    ASSERT_EQ(t.type[3], nvinfer1::DimensionType::kSPATIAL);
}

TEST_F(ChainerTRT_HPP_Test, MakeDimsWith8ItemsShouldBeOK) {
    auto t = chainer_trt::internal::make_dims(1, 2, 3, 4, 5, 6, 7, 8);
    ASSERT_EQ(t.nbDims, 8);
}

TEST_F(ChainerTRT_HPP_Test, MakeDimsWithMoreThan8ItemsCannotBeCompiled) {
    // This code causes compile error
    // make_dims(1, 2, 3, 4, 5, 6, 7, 8, 9);
}

TEST_F(ChainerTRT_HPP_Test, MakeDimsWithNonIntValuesCannotBeCompiled) {
    // This code causes compile error
    // make_dims(1, 2, 3, "hello world");
}

TEST_F(ChainerTRT_HPP_Test, CheckInputOutputExistenceByName) {
    // x = Variable(np.random.randn(1, 3, 16, 16).astype(np.float32))
    // with chainer.using_config('train', False), RetainHook():
    //     y = x + 1
    // retriever = ModelRetriever(out)
    // retriever.register_inputs(x, name='named_input')
    // retriever(y, name='named_out').save()
    const std::string fixture = "test/fixtures/chainer_trt/named_net";
    auto m = chainer_trt::model::build_fp32(fixture, 2, 1);

    ASSERT_EQ((int)m->get_input_names().size(), 1);
    ASSERT_EQ(m->get_input_names()[0], "named_input");

    ASSERT_EQ((int)m->get_output_names().size(), 1);
    ASSERT_EQ(m->get_output_names()[0], "named_out");

    ASSERT_TRUE(m->has_input("named_input"));
    ASSERT_TRUE(m->has_output("named_out"));

    ASSERT_FALSE(m->has_input("named_out"));
    ASSERT_FALSE(m->has_input("fooooobarrrrrrrr"));
    ASSERT_FALSE(m->has_output("named_input"));
    ASSERT_FALSE(m->has_output("foooooobarrrrrrr"));

    assert_vector_eq(m->get_input_dimensions("named_input"),
                     std::vector<int>{3, 16, 16});
    assert_vector_eq(m->get_output_dimensions("named_out"),
                     std::vector<int>{3, 16, 16});

    ASSERT_THROW(m->get_input_dimensions("named_out"), std::invalid_argument);
    ASSERT_THROW(m->get_output_dimensions("named_input"),
                 std::invalid_argument);
}

TEST_F(ChainerTRT_HPP_Test, GetInputOutput) {
    // x = Variable(np.random.randn(1, 3, 16, 16).astype(np.float32))
    // with chainer.using_config('train', False), RetainHook():
    //     y = x + 1
    // retriever = ModelRetriever(out)
    // retriever.register_inputs(x, name='named_input')
    // retriever(y, name='named_out').save()
    const int model_batch_size = 4, infer_batch_size = 2;
    const std::string fixture = "test/fixtures/chainer_trt/named_net";
    auto m = chainer_trt::model::build_fp32(fixture, 2, model_batch_size);
    chainer_trt::infer rt(m);
    chainer_trt::buffer buf(m, infer_batch_size);

    // Send data through named input (an array initialized by 10)
    const int size_per_batch = 3 * 16 * 16;
    const int size_all_batch = infer_batch_size * size_per_batch;
    std::vector<float> in_cpu(size_all_batch, 10.0f);
    cudaMemcpy(buf.get_input("named_input"), in_cpu.data(),
               sizeof(float) * size_all_batch, cudaMemcpyHostToDevice);

    // Run inference
    rt(buf);
    std::vector<float> out_cpu(infer_batch_size * 3 * 16 * 16, 0.0f);
    buf.output_device_to_host(std::vector<void*>{(void*)out_cpu.data()});

    // Check values (10+1==11)
    for(float y : out_cpu)
        ASSERT_EQ(y, 11.0f);

    // Check interface
    ASSERT_EQ(buf.get_input_size("named_input"),
              (int)sizeof(float) * size_all_batch);
    ASSERT_EQ(buf.get_input_size("named_input", false),
              (int)sizeof(float) * size_per_batch);
    ASSERT_EQ(buf.get_input_size(0), (int)sizeof(float) * size_all_batch);
    ASSERT_EQ(buf.get_output_size("named_out"),
              (int)sizeof(float) * size_all_batch);
    ASSERT_EQ(buf.get_output_size("named_out", false),
              (int)sizeof(float) * size_per_batch);
    ASSERT_EQ(buf.get_output_size(0), (int)sizeof(float) * size_all_batch);
    ASSERT_NO_THROW(buf.get_input("named_input"));
    ASSERT_NO_THROW(buf.get_input(0));
    ASSERT_NO_THROW(buf.get_output("named_out"));
    ASSERT_NO_THROW(buf.get_output(0));

    ASSERT_ANY_THROW(buf.get_input_size(1));
    ASSERT_ANY_THROW(buf.get_input_size(-1));
    // not to confuse input with outputs
    ASSERT_ANY_THROW(buf.get_input_size("named_out"));
    ASSERT_ANY_THROW(buf.get_input_size("foooooooo"));
    ASSERT_ANY_THROW(buf.get_input(1));
    ASSERT_ANY_THROW(buf.get_input(-1));
    ASSERT_ANY_THROW(buf.get_input("named_out"));
    ASSERT_ANY_THROW(buf.get_input("foooooooo"));

    ASSERT_ANY_THROW(buf.get_output_size(1));
    ASSERT_ANY_THROW(buf.get_output_size(-1));
    // not to confuse output with inputs
    ASSERT_ANY_THROW(buf.get_output_size("named_input"));
    ASSERT_ANY_THROW(buf.get_output_size("foooooooooo"));
    ASSERT_ANY_THROW(buf.get_output(1));
    ASSERT_ANY_THROW(buf.get_output(-1));
    ASSERT_ANY_THROW(buf.get_output("named_input"));
    ASSERT_ANY_THROW(buf.get_output("foooooooooo"));

    // Check values obtained through name
    cudaMemcpy(out_cpu.data(), buf.get_output("named_out"),
               sizeof(float) * size_all_batch, cudaMemcpyDeviceToHost);
    for(float y : out_cpu)
        ASSERT_EQ(y, 11.0f);
}

TEST_F(ChainerTRT_HPP_Test, MakeBindingByDict) {
    // s = (1, 3, 8, 8)
    // x1 = chainer.Variable(np.random.random(s).astype(np.float32))
    // x2 = chainer.Variable(np.random.random(s).astype(np.float32))
    // c = 2 * np.ones(s).astype(np.float32)
    // with chainer.using_config('train', False):
    //     with RetainHook():
    //         y = x1 + x2
    //         y = y * c
    // retriever = ModelRetriever(out)
    // retriever.register_inputs(x1, name="x1")
    // retriever.register_inputs(x2, name="x2")
    // retriever(y, name="out")
    // retriever.save()
    const std::string dir = "test/fixtures/chainer_trt/raw_binding";
    auto m = chainer_trt::model::build_fp32(dir, 1, 1);
    chainer_trt::infer rt(m);

    const int size = 3 * 8 * 8;
    float *in1_gpu, *in2_gpu, *out_gpu;
    cudaMalloc((void**)&in1_gpu, sizeof(float) * size);
    cudaMalloc((void**)&in2_gpu, sizeof(float) * size);
    cudaMalloc((void**)&out_gpu, sizeof(float) * size);
    const auto in1_cpu = load_values<float>(dir + "/in1.csv");
    const auto in2_cpu = load_values<float>(dir + "/in2.csv");
    std::vector<float> out_cpu(size, 0);
    const auto expected_out_cpu = load_values<float>(dir + "/out.csv");
    cudaMemcpy(in1_gpu, in1_cpu.data(), sizeof(float) * size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(in2_gpu, in2_cpu.data(), sizeof(float) * size,
               cudaMemcpyHostToDevice);

    // check valid binding (by name)
    ASSERT_NO_THROW({
        rt.create_bindings({{"x1", (void*)in1_gpu},
                            {"x2", (void*)in2_gpu},
                            {"out", (void*)out_gpu}});
    });

    // check valid binding (by name, including non-used one)
    ASSERT_NO_THROW({
        rt.create_bindings({{"x1", (void*)in1_gpu},
                            {"x2", (void*)in2_gpu},
                            {"out", (void*)out_gpu},
                            {"ghost", nullptr}});
    });

    // check valid binding (pointer vector)
    ASSERT_NO_THROW({
        rt.create_bindings({(void*)in1_gpu, (void*)in2_gpu}, {(void*)out_gpu});
    });

    // check invalid bindings (insufficient case)
    ASSERT_ANY_THROW({
        rt.create_bindings({{"x1", (void*)in1_gpu}, {"x2", (void*)in2_gpu}});
    });
    ASSERT_ANY_THROW({
        rt.create_bindings({{"x1", (void*)in1_gpu}, {"out", (void*)out_gpu}});
    });
    ASSERT_ANY_THROW(
      { rt.create_bindings({(void*)in1_gpu}, {(void*)out_gpu}); });
    ASSERT_ANY_THROW(
      { rt.create_bindings({(void*)in1_gpu}, {(void*)in1_gpu}); });

    // check invalid bindings (too much case)
    ASSERT_ANY_THROW(
      rt.create_bindings({(void*)in1_gpu, nullptr, nullptr}, {(void*)out_gpu}));

    // check getting binding index
    ASSERT_EQ(rt.get_binding_index("x1"), 0);
    ASSERT_EQ(rt.get_binding_index("x2"), 1);
    ASSERT_EQ(rt.get_binding_index("out"), 2);
    ASSERT_ANY_THROW(rt.get_binding_index("ghost"));
    ASSERT_ANY_THROW(rt.get_binding_index("ConstantInput-0"));

    // check outputs
    auto bindings = rt.create_bindings({{"x1", (void*)in1_gpu},
                                        {"x2", (void*)in2_gpu},
                                        {"out", (void*)out_gpu}});
    ASSERT_EQ((int)bindings.size(), 3); // x1, x2, out
    rt(1, bindings);

    cudaMemcpy(out_cpu.data(), out_gpu, sizeof(float) * size,
               cudaMemcpyDeviceToHost);
    assert_vector_eq(out_cpu, expected_out_cpu);
}

TEST_F(ChainerTRT_HPP_Test, InferFromGPUWithName) {
    const std::string dir = "test/fixtures/chainer_trt/raw_binding";
    auto m = chainer_trt::model::build_fp32(dir, 1, 1);
    chainer_trt::infer rt(m);

    const int size = 3 * 8 * 8;
    float *in1_gpu, *in2_gpu, *out_gpu;
    cudaMalloc((void**)&in1_gpu, sizeof(float) * size);
    cudaMalloc((void**)&in2_gpu, sizeof(float) * size);
    cudaMalloc((void**)&out_gpu, sizeof(float) * size);
    const auto in1_cpu = load_values<float>(dir + "/in1.csv");
    const auto in2_cpu = load_values<float>(dir + "/in2.csv");
    std::vector<float> out_cpu(size, 0);
    const auto expected_out_cpu = load_values<float>(dir + "/out.csv");
    cudaMemcpy(in1_gpu, in1_cpu.data(), sizeof(float) * size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(in2_gpu, in2_cpu.data(), sizeof(float) * size,
               cudaMemcpyHostToDevice);

    // Insufficient inputs
    ASSERT_THROW(
      {
          rt(1, {{"x1", in1_gpu}}, {{"out", out_gpu}});
      },
      std::invalid_argument);

    // Unknown input
    ASSERT_THROW(
      {
          rt(1, {{"x1", in1_gpu}, {"xhogehoge", in2_gpu}}, {{"out", out_gpu}});
      },
      std::invalid_argument);

    // Insufficient outputs
    ASSERT_THROW(
      {
          rt(1, {{"x1", in1_gpu}, {"x2", in2_gpu}}, {});
      },
      std::invalid_argument);

    // Unknown output
    ASSERT_THROW(
      {
          rt(1, {{"x1", in1_gpu}, {"x2", in2_gpu}}, {{"outhogehoge", out_gpu}});
      },
      std::invalid_argument);

    // Allow too much inputs (just ignore)
    ASSERT_NO_THROW({
        rt(1, {{"x1", in1_gpu}, {"x2", in2_gpu}, {"x3", in2_gpu}},
           {{"out", out_gpu}});
    });

    // Allow too much outputs (just ignore)
    ASSERT_NO_THROW({
        rt(1, {{"x1", in1_gpu}, {"x2", in2_gpu}},
           {{"out", out_gpu}, {"out3", out_gpu}});
    });

    // OK case
    rt(1, {{"x1", in1_gpu}, {"x2", in2_gpu}}, {{"out", out_gpu}});

    cudaMemcpy(out_cpu.data(), out_gpu, sizeof(float) * size,
               cudaMemcpyDeviceToHost);
    assert_vector_eq(out_cpu, expected_out_cpu);
}

TEST_F(ChainerTRT_HPP_Test, InferFromCPUWithName) {
    const std::string dir = "test/fixtures/chainer_trt/raw_binding";
    auto m = chainer_trt::model::build_fp32(dir, 1, 1);
    chainer_trt::infer rt(m);

    const int size = 3 * 8 * 8;
    const auto in1 = load_values<float>(dir + "/in1.csv");
    const auto in2 = load_values<float>(dir + "/in2.csv");
    std::vector<float> out(size, 0);
    const auto expected_out = load_values<float>(dir + "/out.csv");

    // Insufficient inputs
    ASSERT_THROW(
      {
          rt.infer_from_cpu(1, {{"x1", in1.data()}}, {{"out", out.data()}});
      },
      std::invalid_argument);

    // Unknown input
    ASSERT_THROW(
      {
          rt.infer_from_cpu(1, {{"x1", in1.data()}, {"xhogehoge", in2.data()}},
                            {{"out", out.data()}});
      },
      std::invalid_argument);

    // Insufficient outputs
    ASSERT_THROW(
      {
          rt.infer_from_cpu(1, {{"x1", in1.data()}, {"x2", in2.data()}}, {});
      },
      std::invalid_argument);

    // Unknown output
    ASSERT_THROW(
      {
          rt.infer_from_cpu(1, {{"x1", in1.data()}, {"x2", in2.data()}},
                            {{"outhogehoge", out.data()}});
      },
      std::invalid_argument);

    // Allow too much inputs (just ignore)
    ASSERT_NO_THROW({
        rt.infer_from_cpu(
          1, {{"x1", in1.data()}, {"x2", in2.data()}, {"x3", in2.data()}},
          {{"out", out.data()}});
    });

    // Allow too much outputs (just ignore)
    ASSERT_NO_THROW({
        rt.infer_from_cpu(1, {{"x1", in1.data()}, {"x2", in2.data()}},
                          {{"out", out.data()}, {"out3", out.data()}});
    });

    // OK case
    rt.infer_from_cpu(1, {{"x1", in1.data()}, {"x2", in2.data()}},
                      {{"out", out.data()}});

    assert_vector_eq(out, expected_out);
}

TEST_F(ChainerTRT_HPP_Test, InferFromCPUWithNameUsingBuffer) {
    const std::string dir = "test/fixtures/chainer_trt/raw_binding";
    auto m = chainer_trt::model::build_fp32(dir, 1, 1);
    chainer_trt::infer rt(m);

    const int size = 3 * 8 * 8;
    const auto in1 = load_values<float>(dir + "/in1.csv");
    const auto in2 = load_values<float>(dir + "/in2.csv");
    std::vector<float> out(size, 0);
    const auto expected_out = load_values<float>(dir + "/out.csv");

    auto buffer = rt.create_buffer(1);

    buffer->input_host_to_device({{"x1", in1.data()}, {"x2", in2.data()}});
    rt(*buffer);
    buffer->output_device_to_host({{"out", out.data()}});

    assert_vector_eq(out, expected_out);
}
