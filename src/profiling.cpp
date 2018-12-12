/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <iomanip>
#include <iostream>

#include <nvToolsExt.h>

#include "chainer_trt/profiling.hpp"

namespace chainer_trt {
// Needed to declear instance.
// See also: http://faithandbrave.hateblo.jp/entry/2017/10/16/160146
constexpr uint32_t __NVTXHook::colors[];

__NVTXHook::__NVTXHook(const char* name) {
    nvtxRangePushA(name);
}

__NVTXHook::__NVTXHook(const char* name, int color_id) {
    // color_id % n_colors (with allowing neg color_id).
    // cf. https://stackoverflow.com/questions/1907565
    const int n_colors = sizeof(__NVTXHook::colors) / sizeof(uint32_t);
    color_id = (color_id % n_colors + n_colors) % n_colors;

    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = __NVTXHook::colors[color_id];
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;

    nvtxRangePushEx(&eventAttrib);
}

__NVTXHook::~__NVTXHook() {
    nvtxRangePop();
}

void default_profiler::reportLayerTime(const char* layer_name, float ms) {
    if(records.find(layer_name) == records.end()) {
        records[layer_name] = 0;
        orders.push_back(records.find(layer_name));
    }
    records[layer_name] += ms;
    call_counts[layer_name] += 1;
}

void default_profiler::show_profiling_result(std::ostream& ost,
                                             const std::string& mode) const {
    // Get some statistics
    unsigned int max_length = 0;
    double ms_sum = 0;
    for(const auto& rec : orders) {
        max_length = std::max((unsigned int)rec->first.size(), max_length);
        ms_sum += rec->second;
    }

    if(mode == "md") {
        // prettified "| Layer name | #call | total ms | ms/call | % |"
        ost << "| " << std::left << std::setw(max_length) << "Layer name";
        ost << " | " << std::right << std::setw(5) << "#call";
        ost << " | " << std::right << std::setw(10) << "total ms";
        ost << " | " << std::right << std::setw(8) << "ms/call";
        ost << " | " << std::right << std::setw(8) << "%";
        ost << " |" << std::endl;

        // clang-format off
            ost << "|:"; for(unsigned int i = 0; i <= max_length; ++i) ost << '-';
            ost << "|:"; for(int i = 0; i <= 5; ++i) ost << '-';
            ost << "|:"; for(int i = 0; i <= 10; ++i) ost << '-';
            ost << "|:"; for(int i = 0; i <= 8; ++i) ost << '-';
            ost << "|:"; for(int i = 0; i <= 8; ++i) ost << '-';
            ost << "|" << std::endl;
        // clang-format on

        for(const auto rec : orders) {
            const int n_call = call_counts.find(rec->first)->second;
            ost << "| " << std::left << std::setw(max_length) << rec->first;
            ost << " | " << std::right << std::setw(5) << n_call;
            ost << " | " << std::right << std::fixed << std::setprecision(5)
                << std::setw(10) << rec->second;
            ost << " | " << std::right << std::fixed << std::setprecision(5)
                << std::setw(8) << (rec->second / n_call);
            ost << " | " << std::right << std::fixed << std::setprecision(3)
                << std::setw(7) << (100 * rec->second / ms_sum);
            ost << "% |" << std::endl;
        }
    } else if(mode == "csv") {
        ost << "layer_name,#call,total ms,ms/call,%" << std::endl;

        for(const auto rec : orders) {
            const int n_call = call_counts.find(rec->first)->second;
            ost << "\"" << rec->first << "\"," << n_call << ",";
            ost << rec->second << "," << (rec->second / n_call);
            ost << "," << (100 * rec->second / ms_sum) << std::endl;
        }
    }
}

void default_profiler::reset() {
    records.clear();
    orders.clear();
    call_counts.clear();
}
}
