# Copyright (c) 2017 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

FILE(GLOB_RECURSE TEST_SOURCES *.cpp)
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/test)
add_executable(test_chainer_trt ${TEST_SOURCES})
target_link_libraries(test_chainer_trt gtest pthread cudart chainer_trt glog)
FILE(COPY fixtures DESTINATION ${CMAKE_BINARY_DIR}/test)
