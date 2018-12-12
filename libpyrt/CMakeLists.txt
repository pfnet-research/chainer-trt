# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

cmake_minimum_required(VERSION 2.8)
project(chainer-trt)


CHECK_INCLUDE_FILE_CXX("Python.h" HAVE_PYTHON_H)
IF(NOT HAVE_PYTHON_H)
  message( FATAL_ERROR "Python.h Not Found" )
endif()

CHECK_INCLUDE_FILE_CXX("pybind11/pybind11.h" HAVE_PYBIND11)
IF(NOT HAVE_PYBIND11)
  message( FATAL_ERROR "pybind11 Not Found" )
endif()

#find_package(PythonLibs 3.6 REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})
file(GLOB PYINFER_SOURCES libpyrt.cpp)
add_library(pyrt SHARED ${PYINFER_SOURCES})
target_link_libraries(pyrt chainer_trt cudart nvinfer glog)
install(TARGETS pyrt DESTINATION lib)

