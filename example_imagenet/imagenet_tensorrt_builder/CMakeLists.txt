# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)

add_executable(imagenet_tensorrt_builder imagenet_tensorrt_builder.cpp cmdline.c)
target_link_libraries(imagenet_tensorrt_builder cudart nvinfer chainer_trt glog ${OPENCV_LIBS})
