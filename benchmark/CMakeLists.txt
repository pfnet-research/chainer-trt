# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)

FILE(GLOB_RECURSE BENCHMARK_SOURCES *.cpp)
add_executable(bench ${BENCHMARK_SOURCES})

if(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
include_directories(../../prebuilt_qnx/glog/include ../../prebuilt_qnx/benchmark/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../prebuilt_qnx/glog/lib ${CMAKE_CURRENT_SOURCE_DIR}/../../prebuilt_qnx/benchmark/lib)
endif()

find_library(CUDART cudart HINTS ENV LIBRARY_PATH)
find_library(NVINFER nvinfer HINTS ENV LIBRARY_PATH)
find_library(GLOG glog HINTS LIBRARY_PATH)
find_library(BENCHMARK benchmark HINTS LIBRARY_PATH)
set(LINK_LIBS ${CUDART} ${NVINFER} ${GLOG} ${BENCHMARK})

if(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
find_library(NVMEDIA nvmedia HINTS ENV LIBRARY_PATH)
list(APPEND LINK_LIBS ${NVMEDIA})
endif()

target_link_libraries(bench ${LINK_LIBS} chainer_trt)
