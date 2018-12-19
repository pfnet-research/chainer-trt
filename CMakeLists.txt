# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)
project(chainer-trt)

option(WITH_TEST "Build test" ON)
option(WITH_TOOLS "Build tools" ON)
option(WITH_PYTHON_LIB "Build Python interface" ON)
option(WITH_BENCHMARK "Build micro-benchmarks" OFF)
option(WITH_EXAMPLES "Build ImageNet examples" ON)

find_package(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall -Wextra -Wshadow -Wendif-labels -Wformat-security -Wformat-zero-length -Wcast-align")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O3 -arch=sm_60")

option(WITH_NVTX "Enable NVTX profiling hook" OFF)
if(WITH_NVTX)
    add_definitions(-DWITH_NVTX)
endif()

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src")

# Check libraries
include(CheckIncludeFileCXX)

CHECK_INCLUDE_FILE_CXX("NvInfer.h" HAVE_TENSORRT)
IF(NOT HAVE_TENSORRT)
  message( FATAL_ERROR "TensorRT Not Found" )
endif()


FILE(GLOB CUDA_SOURCES src/cuda/*.cu)
FILE(GLOB_RECURSE LIBRARY_SOURCES src/*.cpp)

find_library(CUDART cudart HINTS ENV LIBRARY_PATH)
find_library(NVINFER nvinfer HINTS ENV LIBRARY_PATH)
find_library(NVTOOLSEXT nvToolsExt HINTS ENV LIBRARY_PATH)
set(LINK_LIBS ${CUDART} ${NVINFER} ${NVTOOLSEXT})

# Shared lib
cuda_add_library(chainer_trt SHARED ${LIBRARY_SOURCES} ${CUDA_SOURCES})
set_target_properties(chainer_trt PROPERTIES VERSION 1.0)
target_link_libraries(chainer_trt ${LINK_LIBS})

# Static lib (with the same name)
# (c.f. https://cmake.org/Wiki/CMake_FAQ#How_do_I_make_my_shared_and_static_libraries_have_the_same_root_name.2C_but_different_suffixes.3F )
cuda_add_library(chainer_trt_a ${LIBRARY_SOURCES} ${CUDA_SOURCES})
SET_TARGET_PROPERTIES(chainer_trt_a PROPERTIES OUTPUT_NAME "chainer_trt")
target_link_libraries(chainer_trt_a ${LINK_LIBS})

install(DIRECTORY include/chainer_trt DESTINATION include)
install(TARGETS chainer_trt LIBRARY DESTINATION lib)
install(TARGETS chainer_trt_a ARCHIVE DESTINATION lib)


# Build subtools
if(WITH_PYTHON_LIB)
    add_subdirectory(libpyrt)
endif()

if(WITH_TOOLS)
    add_subdirectory(tools)
endif()

if(WITH_TEST)
    add_subdirectory(test)
endif()

if(WITH_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(WITH_EXAMPLES)
    add_subdirectory(example_imagenet)
endif()

if(WITH_EXAMPLES)
    add_subdirectory(example_external_plugin)
endif()
