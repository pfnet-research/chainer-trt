/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <memory>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/external/picojson_helper.hpp"
#include "chainer_trt/plugin.hpp"

#include "../include/plugins/argmax.hpp"
#include "../include/plugins/broadcast_to.hpp"
#include "../include/plugins/constant_elementwise.hpp"
#include "../include/plugins/directed_pooling.hpp"
#include "../include/plugins/leaky_relu.hpp"
#include "../include/plugins/resize.hpp"
#include "../include/plugins/resize_argmax.hpp"
#include "../include/plugins/shift.hpp"
#include "../include/plugins/slice.hpp"
#include "../include/plugins/sum.hpp"
#include "../include/plugins/where.hpp"

namespace chainer_trt {
namespace plugin {

    static bool str_match(const char* str, const char* kw) {
        return std::string(str).find(kw) != std::string::npos;
    }

    static std::string get_type(const char* layerName) {
        // Given "GetItem-0-1", p -> 7
        std::string name(layerName);
        auto p = name.find("-");
        if(p == std::string::npos || p == 0) {
            std::ostringstream oss;
            oss << "It seems layer name \"" << layerName << "\" ";
            oss << "does not include layer type. chainer-trt assumes layer ";
            oss << "name to be \"type-depth-number\".";
            throw std::runtime_error(oss.str());
        }

        return name.substr(0, p);
    }

    plugin_factory::plugin_factory() {
        add_builder_deserializer("Shift", shift::build_layer,
                                 shift::deserialize);
        add_builder_deserializer("ArgMax", argmax::build_layer,
                                 argmax::deserialize);
    }

    nvinfer1::ILayer* plugin_factory::build_plugin(
      network_def network, const picojson::object& layer_params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names,
      const std::string& model_dir) {
        auto type = param_get<std::string>(layer_params, "type");
        auto f = plugin_builders[type];
        return f(network, layer_params, dt, tensor_names, model_dir);
    }

    nvinfer1::IPlugin* plugin_factory::createPlugin(const char* layerName,
                                                    const void* buf,
                                                    size_t len) {
        std::string type = get_type(layerName);
        if(is_registered(type))
            return plugin_deserializers[type](buf, len);

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
        else if(str_match(layerName, "DirectedPooling") or
                str_match(layerName, "CornerPooling"))
            return new directed_pooling(buf, len);
        else if(str_match(layerName, "ResizeArgmax"))
            return new resize_argmax(buf, len);
        else if(str_match(layerName, "Resize"))
            return new resize(buf, len);
        else if(str_match(layerName, "BroadcastTo"))
            return new broadcast_to(buf, len);
        else if(str_match(layerName, "Sum"))
            return new sum(buf, len);
        else if(str_match(layerName, "Where"))
            return new where(buf, len);

        throw layer_not_implemented(layerName, "Unknown");
    }
}
}
