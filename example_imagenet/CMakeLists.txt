# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

configure_file(labels.txt labels.txt COPYONLY)

add_subdirectory(imagenet_tensorrt_builder)
add_subdirectory(imagenet_infer)
add_subdirectory(imagenet_infer_fast)
