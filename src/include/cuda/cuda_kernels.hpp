/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#include <cuda_fp16.h>

namespace chainer_trt {
namespace plugin {
    // Applies slice operations on GPU, implemented in slice.cu
    template <typename T>
    void apply_slice(const T* src_gpu, T* dest_gpu, int* mapping_gpu, int n_src,
                     int n_dst, int batch_size, cudaStream_t stream);

    // Interface to the CUDA kernel of constant elementwise arithmetics
    template <typename T>
    void apply_eltw_sum(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream);
    template <typename T>
    void apply_eltw_sub(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream);
    template <typename T>
    void apply_eltw_mul(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream);
    template <typename T>
    void apply_eltw_div(const T* src_gpu, int n_in, const T* vals_gpu,
                        int n_values, T* dst_gpu, int batch_size,
                        cudaStream_t stream);

    template <typename T>
    void apply_leaky_relu(const T*, T*, int, float, int, cudaStream_t);

    template <typename T>
    void apply_shift(const T* src_gpu, int batch_size, int c, int h, int w,
                     int kh, int kw, int dy, int dx, int grid_size,
                     int block_size, T* dst_gpu, cudaStream_t stream);

    // Applies multidimensional transpose
    // This layer is only for TensorRT 3.x, so no need to support fp16
    void apply_transpose(const float* d_src, float* d_dst, int* d_indexes,
                         int in_size, int batch_size, cudaStream_t stream);

    // Initializes multidimensional transpose parameters
    void initialize_transpose_indexes(int* d_dst, int* i_strides, int* shuffle,
                                      int* i_d, int* o_strides, int in_size,
                                      int id_size);

    // Applies broadcast_to operation on GPU, implemented in broadcast_to.cu
    template <typename T>
    void apply_broadcast_to(const T* d_src, T* d_dst, int* d_i_strides,
                            int* d_o_strides, int in_size, int out_size,
                            int nb_dims, int batch_size, cudaStream_t stream);

    // Applies image resize operation
    template <typename T>
    void apply_resize(const T* src, T* dst, int batch_size, int c, int in_h,
                      int in_w, int out_h, int out_w, cudaStream_t stream);

    // Applies argmax on image with shape CHW
    template <typename T>
    void apply_argmax(const T* d_src, T* d_dst, int channel, int stride,
                      int in_size, int batch_size, cudaStream_t stream);

    // Applies image resize and argmax operation
    template <typename T>
    void apply_resize_argmax(const T* src, T* dst, int batch_size, int c,
                             int in_h, int in_w, int out_h, int out_w,
                             cudaStream_t stream);

    // Applies sum
    template <typename T>
    void apply_sum(const T* d_src, T* d_dst, int channel, int stride,
                   int in_size, int batch_size, cudaStream_t stream);

    // Applies where operation
    template <typename T>
    void apply_where(const T* a, const T* b, const T* c, T* dst, int n_in,
                     int batch_size, cudaStream_t stream);

    template <typename T>
    void apply_directed_pooling(const T* src_gpu, int B, int C, int H, int W,
                                bool horizontal, bool rev, T* dst_gpu,
                                cudaStream_t stream);

    // Check if width and height are supported
    bool is_supported_size_by_resize_argmax(int in_h, int in_w, int out_h,
                                            int out_w);
}
}
