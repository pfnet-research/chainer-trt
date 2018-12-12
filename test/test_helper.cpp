/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>
#include <vector>

#include "chainer_trt/chainer_trt.hpp"
#include "include/chainer_trt_impl.hpp"
#include "test_helper.hpp"

template <>
std::vector<float> load_values(const std::string &filename) {
    std::vector<float> vals;
    std::ifstream ifs(filename);
    if(!ifs)
        throw std::invalid_argument("File " + filename + " cannot be opened.");
    std::string token;
    while(std::getline(ifs, token, ','))
        vals.push_back(std::stof(token));
    return vals;
}

template <>
std::vector<__half> load_values(const std::string &filename) {
    const std::vector<float> vals = load_values<float>(filename);
    std::vector<__half> ret(vals.size());
    chainer_trt::internal::float2half(vals.data(), ret.data(), ret.size());
    return ret;
}

template <>
void assert_vector_eq(const std::vector<__half> &vec1,
                      const std::vector<__half> &vec2) {
    ASSERT_EQ(vec1.size(), vec2.size());
    const int n = vec1.size();
    std::vector<float> vec1_f(n);
    std::vector<float> vec2_f(n);
    chainer_trt::internal::half2float(vec1.data(), vec1_f.data(), n);
    chainer_trt::internal::half2float(vec2.data(), vec2_f.data(), n);
    assert_vector_eq(vec1_f, vec2_f);
}

template <>
std::vector<float> load_values_binary(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);

    // get filesize
    int fsize = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    fsize = static_cast<int>(ifs.tellg()) - fsize;
    ifs.seekg(0);

    std::vector<float> ret(fsize / sizeof(float));
    ifs.read(reinterpret_cast<char *>(ret.data()), ret.size() * sizeof(float));
    return ret;
}

template <>
std::vector<__half> load_values_binary(const std::string &filename) {
    std::vector<float> raw = load_values_binary<float>(filename);
    std::vector<__half> ret(raw.size());
    chainer_trt::internal::float2half(raw.data(), ret.data(), ret.size());
    return ret;
}

void assert_eq(float t1, float t2) {
    ASSERT_FLOAT_EQ(t1, t2);
}

void assert_eq(int t1, int t2) {
    ASSERT_EQ(t1, t2);
}

void assert_dims_eq(const nvinfer1::Dims &dim1, const nvinfer1::Dims &dim2) {
    ASSERT_EQ(dim1.nbDims, dim2.nbDims);
    for(int i = 0; i < dim1.nbDims; ++i)
        ASSERT_EQ(dim1.d[i], dim2.d[i]);
}

void assert_vector_near(const std::vector<float> &vec1,
                        const std::vector<float> &vec2, float abs_error) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for(unsigned i = 0; i < vec1.size(); ++i)
        EXPECT_NEAR(vec1[i], vec2[i], abs_error);
}

void assert_vector_near(const std::vector<__half> &vec1,
                        const std::vector<__half> &vec2, float abs_error) {
    ASSERT_EQ(vec1.size(), vec2.size());
    const int n = vec1.size();
    std::vector<float> vec1_f(n);
    std::vector<float> vec2_f(n);
    chainer_trt::internal::half2float(vec1.data(), vec1_f.data(), n);
    chainer_trt::internal::half2float(vec2.data(), vec2_f.data(), n);
    assert_vector_near(vec1_f, vec2_f, abs_error);
}
