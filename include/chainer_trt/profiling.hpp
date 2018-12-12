/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <NvInfer.h>
#pragma GCC diagnostic pop

#include <iostream>
#include <map>
#include <vector>

namespace chainer_trt {
// Note: this class shouldn't be inside #ifdef WITH_NVTX
// Because even if chainer-trt is built without -DWITH_NVTX,
// user code should still be able to use nvtx_profile
class __NVTXHook {
    // Randomly shuffled color definition
    static constexpr uint32_t colors[] = {
      0x00800080, 0x00000080, 0x0000ffff, 0x00c0c0c0, 0x00ff0000,
      0x00008000, 0x00808000, 0x000000ff, 0x0000ff00, 0x00008080,
      0x00800000, 0x00ff00ff, 0x00ffff00,
    };

public:
    __NVTXHook(const char* name);
    __NVTXHook(const char* name, int color_id);
    ~__NVTXHook();
    operator bool() const { return 1; } // Just for nvtx_profile macro
};
}

#ifdef WITH_NVTX

#define nvtx_profile(name) \
    if(chainer_trt::__NVTXHook __nvtx_hook = chainer_trt::__NVTXHook(name))

#define nvtx_profile_color(name, color_id)   \
    if(chainer_trt::__NVTXHook __nvtx_hook = \
         chainer_trt::__NVTXHook(name, color_id))

#else

#define nvtx_profile(name)
#define nvtx_profile_color(name, color_id)

#endif

namespace chainer_trt {
struct default_profiler : public nvinfer1::IProfiler {
    std::map<std::string, double> records;
    std::vector<std::map<std::string, double>::const_iterator> orders;
    std::map<std::string, int> call_counts;

    void reportLayerTime(const char* layer_name, float ms) override;

    void show_profiling_result(std::ostream& ost,
                               const std::string& mode) const;
    void reset();
};
}
