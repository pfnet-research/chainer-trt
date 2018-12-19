/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */
#pragma once

#include <sstream>
#include <string>

#include "chainer_trt/external/picojson.h"

namespace chainer_trt {
template <typename T>
T param_get(const picojson::object& params, const std::string& key) {
    auto p = params.find(key);
    if(p == params.end()) {
        std::ostringstream oss;
        oss << "Error: key \"" << key << "\" is not found in the ";
        oss << "parameter dictionary ";
        oss << picojson::value(params).serialize();
        throw std::runtime_error(oss.str());
    }
    return p->second.get<T>();
}

template <>
int param_get<int>(const picojson::object& params, const std::string& key);

template <>
float param_get<float>(const picojson::object& params, const std::string& key);
}
