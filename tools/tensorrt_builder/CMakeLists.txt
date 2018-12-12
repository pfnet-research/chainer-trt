# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

add_executable(tensorrt_builder tensorrt_builder.cpp cmdline.c)
target_link_libraries(tensorrt_builder cudart nvinfer chainer_trt glog)
install(TARGETS tensorrt_builder DESTINATION bin)
