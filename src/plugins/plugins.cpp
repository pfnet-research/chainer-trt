/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <memory>

#include "chainer_trt/chainer_trt.hpp"

#include "../include/plugins/plugins.hpp"

namespace chainer_trt {
namespace plugin {

    static bool str_match(const char* str, const char* kw) {
        return std::string(str).find(kw) != std::string::npos;
    }

    nvinfer1::IPlugin* plugin_factory::createPlugin(const char* layerName,
                                                    const void* buf,
                                                    size_t len) {
        // No worries about memory leak.
        // TensorRT 5 releases plugin objects internally (different from 4)
        if(str_match(layerName, "GetItem"))
            return new get_item(buf, len);
        else if(str_match(layerName, "AddConstant") ||
                str_match(layerName, "SubFromConstant") ||
                str_match(layerName, "MulConstant") ||
                str_match(layerName, "DivFromConstant"))
            return new constant_elementwise(buf, len);
        else if(str_match(layerName, "LeakyReLU"))
            return new leaky_relu(buf, len);
        else if(str_match(layerName, "Shift"))
            return new shift(buf, len);
        else if(str_match(layerName, "DirectedPooling") or
                str_match(layerName, "CornerPooling"))
            return new directed_pooling(buf, len);
        else if(str_match(layerName, "ResizeArgmax"))
            return new resize_argmax(buf, len);
        else if(str_match(layerName, "Resize"))
            return new resize(buf, len);
        else if(str_match(layerName, "BroadcastTo"))
            return new broadcast_to(buf, len);
        else if(str_match(layerName, "ArgMax"))
            return new argmax(buf, len);
        else if(str_match(layerName, "Sum"))
            return new sum(buf, len);
        else if(str_match(layerName, "Where"))
            return new where(buf, len);

        throw layer_not_implemented(layerName, "Unknown");
    }
}
}
