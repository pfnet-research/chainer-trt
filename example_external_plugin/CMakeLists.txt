# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

cuda_add_executable(increment_plugin increment_plugin.cpp increment_kernel.cu cmdline.c)
target_link_libraries(increment_plugin cudart nvinfer chainer_trt)
