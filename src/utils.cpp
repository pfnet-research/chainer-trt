/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <iostream>

#include <cuda_runtime_api.h>

#include "chainer_trt/utils.hpp"

void show_device_info(int gpu_id) {
    if(gpu_id == -1)
        cudaGetDevice(&gpu_id);

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, gpu_id);

    std::cout << "Using GPU=" << gpu_id << " ";
    std::cout << "(Name=\"" << dev.name << "\",";
    std::cout << "CC=" << dev.major << "." << dev.minor << ", ";
    std::cout << "VRAM=" << (dev.totalGlobalMem >> 20) << "MB";
    std::cout << ")" << std::endl;
}

void hwc2chw(float* dst, const float* src, int w, int h) {
    static const int n_c = 3;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
            for(int c = 0; c < n_c; ++c)
                dst[(c * h + y) * w + x] = src[(y * w + x) * n_c + c];
}
