# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)

add_executable(imagenet_infer imagenet_infer.cpp cmdline.c)
target_link_libraries(imagenet_infer cudart nvinfer chainer_trt glog ${OPENCV_LIBS})
