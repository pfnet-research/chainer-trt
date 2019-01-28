/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <array>
#include <cassert>
#include <fstream>
#include <mutex>
#include <queue>
#include <stdexcept>

#include <cuda_fp16.h>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/external/picojson.h"
#include "chainer_trt/external/picojson_helper.hpp"
#include "include/chainer_trt_impl.hpp"

namespace chainer_trt {

using network_def = std::shared_ptr<nvinfer1::INetworkDefinition>;
using name_tensor_map = std::map<std::string, nvinfer1::ITensor*>;

using chainer_trt::param_get;

// ctor
model::model() {}

namespace internal {
    void build_input(network_def network, const picojson::object& params,
                     name_tensor_map& tensor_names) {
        auto name = param_get<std::string>(params, "name");
        auto shape = param_get<picojson::array>(params, "shape");

        nvinfer1::Dims dims = shapes_to_dims(shape);

        tensor_names[name] =
          network->addInput(name.c_str(), nvinfer1::DataType::kFLOAT, dims);
    }

    nvinfer1::ILayer*
    build_constant_input(network_def network, const picojson::object& params,
                         const std::string& model_dir,
                         std::shared_ptr<internal::weights_manager> weights) {
        auto input_tensor = param_get<std::string>(params, "input_tensor");
        auto shape = param_get<picojson::array>(params, "shape");

        nvinfer1::Weights w = weights->load_weights_as(
          model_dir + "/" + input_tensor, nvinfer1::DataType::kFLOAT);
        return network->addConstant(shapes_to_dims(shape), w);
    }

    nvinfer1::ILayer* build_linear_interpolate(
      network_def network, const picojson::object& params,
      nvinfer1::DataType dt, const name_tensor_map& tensor_names) {
        (void)dt;

        auto source_elements = param_get<picojson::array>(params, "sources");
        std::vector<nvinfer1::ITensor*> inputs;
        for(picojson::value source_element : source_elements) {
            const std::string& source = source_element.get<std::string>();
            auto source_tensor = tensor_names.find(source);
            if(source_tensor == tensor_names.end())
                return NULL;
            inputs.push_back(source_tensor->second);
        }
        assert(inputs.size() == 3);

        // px+(1-p)y
        // =px-py+y
        // =p(x-y)+y
        // here, p=inputs[0], x=inputs[1], y=inputs[2]
        const auto x_minus_y = network->addElementWise(
          *inputs[1], *inputs[2], nvinfer1::ElementWiseOperation::kSUB);
        const auto p_x_minus_y =
          network->addElementWise(*x_minus_y->getOutput(0), *inputs[0],
                                  nvinfer1::ElementWiseOperation::kPROD);
        return network->addElementWise(*p_x_minus_y->getOutput(0), *inputs[2],
                                       nvinfer1::ElementWiseOperation::kSUM);
    }

    nvinfer1::ILayer*
    build_conv(network_def network, const picojson::object& params,
               nvinfer1::DataType dt, const name_tensor_map& tensor_names,
               const std::string& model_dir,
               std::shared_ptr<internal::weights_manager> weights) {
        auto source = param_get<std::string>(params, "source");
        const int kernel_w = param_get<int>(params, "kernel_w");
        const int kernel_h = param_get<int>(params, "kernel_h");
        const int pad_w = param_get<int>(params, "pad_w");
        const int pad_h = param_get<int>(params, "pad_h");
        const int stride_x = param_get<int>(params, "stride_x");
        const int stride_y = param_get<int>(params, "stride_y");
        const int n_out = param_get<int>(params, "n_out");
        const auto kernel_file =
          param_get<std::string>(params, "kernel_weights_file");
        const auto bias_file =
          param_get<std::string>(params, "bias_weights_file");

        const int groups = params.find("groups") != params.end()
                             ? param_get<int>(params, "groups")
                             : 1;
        const int dilation_y = params.find("dilation_y") != params.end()
                                 ? param_get<int>(params, "dilation_y")
                                 : 1;
        const int dilation_x = params.find("dilation_x") != params.end()
                                 ? param_get<int>(params, "dilation_x")
                                 : 1;

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::DimsHW dim = {kernel_h, kernel_w};
        nvinfer1::Weights kernel_weight =
          weights->load_weights_as(model_dir + "/" + kernel_file, dt);
        nvinfer1::Weights bias_weight =
          weights->load_weights_as(model_dir + "/" + bias_file, dt);
        nvinfer1::IConvolutionLayer* conv = network->addConvolution(
          *input, n_out, dim, kernel_weight, bias_weight);
        conv->setPadding(nvinfer1::DimsHW{pad_h, pad_w});
        conv->setStride(nvinfer1::DimsHW{stride_y, stride_x});
        conv->setNbGroups(groups);
        conv->setDilation(nvinfer1::DimsHW{dilation_y, dilation_x});
        return conv;
    }

    nvinfer1::ILayer*
    build_deconv(network_def network, const picojson::object& params,
                 nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                 const std::string& model_dir,
                 std::shared_ptr<internal::weights_manager> weights) {
        auto source = param_get<std::string>(params, "source");
        const int kernel_w = param_get<int>(params, "kernel_w");
        const int kernel_h = param_get<int>(params, "kernel_h");
        const int pad_w = param_get<int>(params, "pad_w");
        const int pad_h = param_get<int>(params, "pad_h");
        const int stride_x = param_get<int>(params, "stride_x");
        const int stride_y = param_get<int>(params, "stride_y");
        const int n_out = param_get<int>(params, "n_out");
        const auto kernel_file =
          param_get<std::string>(params, "kernel_weights_file");
        const auto bias_file =
          param_get<std::string>(params, "bias_weights_file");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        const int groups = params.find("groups") != params.end()
                             ? param_get<int>(params, "groups")
                             : 1;

        // Hmm?  No nead to deal with cover_all with deconv_outdim_formula??

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::DimsHW dim = {kernel_h, kernel_w};
        nvinfer1::Weights kernel_weight =
          weights->load_weights_as(model_dir + "/" + kernel_file, dt);
        nvinfer1::Weights bias_weight =
          weights->load_weights_as(model_dir + "/" + bias_file, dt);
        nvinfer1::IDeconvolutionLayer* deconv = network->addDeconvolution(
          *input, n_out, dim, kernel_weight, bias_weight);
        deconv->setPadding(nvinfer1::DimsHW{pad_h, pad_w});
        deconv->setStride(nvinfer1::DimsHW{stride_y, stride_x});
        deconv->setNbGroups(groups);
        return deconv;
    }

    nvinfer1::ILayer*
    build_bn(network_def network, const picojson::object& params,
             nvinfer1::DataType dt, name_tensor_map& tensor_names,
             const std::string& model_dir,
             std::shared_ptr<internal::weights_manager> weights) {
        auto source = param_get<std::string>(params, "source");
        auto mean_f = param_get<std::string>(params, "mean_weights_file");
        auto var_f = param_get<std::string>(params, "var_weights_file");
        auto gamma_f = param_get<std::string>(params, "gamma_weights_file");
        auto beta_f = param_get<std::string>(params, "beta_weights_file");
        const float eps = param_get<float>(params, "eps");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        constexpr auto fp32 = nvinfer1::DataType::kFLOAT;
        auto mean = weights->load_weights_as(model_dir + "/" + mean_f, fp32);
        auto var = weights->load_weights_as(model_dir + "/" + var_f, fp32);
        auto gamma = weights->load_weights_as(model_dir + "/" + gamma_f, fp32);
        auto beta = weights->load_weights_as(model_dir + "/" + beta_f, fp32);
        nvinfer1::Weights none{dt, NULL, 0};

        // BN operation https://arxiv.org/pdf/1502.03167.pdf
        // (see definition of chainer BatchNormalizationFunction#forward
        //  (numpy part) in batch_normalization.py)
        // given: x, mean, var, gamma, beta
        // stdev = sqrt(var + eps)
        // x_hat = (x - mean) / stdev       //normalization
        // y     = gamma * x_hat + beta
        //       = gamma * (x - mean) / stdev + beta
        //       = gamma * x / stdev - gamma * mean / stdev + beta
        //       = gamma * x / stdev - (gamma * mean / stdev - beta)
        //       = gamma * x / sqrt(var+eps) - (gamma * mean / sqrt(var+eps) -
        //         beta)
        // here,
        // gamma = gamma / sqrt(var + eps)
        // beta  = -(gamma * mean / sqrt(var + eps) - beta)
        // therefore,
        // y     = gamma * x + beta    // this can be computer by a scale layer

        float* gamma_buf = (float*)gamma.values;
        float* beta_buf = (float*)beta.values;
        float* mean_buf = (float*)mean.values;
        float* var_buf = (float*)var.values;
        for(int i = 0; i < gamma.count; ++i) {
            beta_buf[i] =
              -((gamma_buf[i] * mean_buf[i]) / std::sqrt(var_buf[i] + eps) -
                beta_buf[i]);
            gamma_buf[i] = gamma_buf[i] / std::sqrt(var_buf[i] + eps);
        }

        if(dt == nvinfer1::DataType::kHALF) {
            const int n = beta.count;
            float2half((const float*)beta.values, (__half*)beta.values, n);
            float2half((const float*)gamma.values, (__half*)gamma.values, n);

            // beta and gamma are originally created as kFLOAT Weights,
            // so they need to be reinterpreted as kHALF weights
            beta.type = nvinfer1::DataType::kHALF;
            gamma.type = nvinfer1::DataType::kHALF;
        }

        nvinfer1::ITensor* input = source_tensor->second;
        return network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL, beta,
                                 gamma, none);
    }

    nvinfer1::ILayer* build_lrn(network_def network,
                                const picojson::object& params,
                                const name_tensor_map& tensor_names) {
        const auto source = param_get<std::string>(params, "source");
        const int window = param_get<int>(params, "n");
        const float alpha = param_get<float>(params, "alpha");
        const float beta = param_get<float>(params, "beta");
        const float k = param_get<float>(params, "k");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        return network->addLRN(*input, window, alpha, beta, k);
    }

    nvinfer1::ILayer* build_matmul(network_def network,
                                   const picojson::object& params,
                                   const name_tensor_map& tensor_names) {
        const auto sources = param_get<picojson::array>(params, "sources");
        const bool transa = param_get<bool>(params, "transa");
        const bool transb = param_get<bool>(params, "transb");

        std::vector<nvinfer1::ITensor*> inputs;
        for(picojson::value source_element : sources) {
            const std::string& source = source_element.get<std::string>();
            auto source_tensor = tensor_names.find(source);
            if(source_tensor == tensor_names.end())
                return NULL;
            inputs.push_back(source_tensor->second);
        }

        assert(inputs.size() == 2);
        return network->addMatrixMultiply(*inputs[0], transa, *inputs[1],
                                          transb);
    }

    nvinfer1::ILayer* build_eltw(network_def network,
                                 const picojson::object& params,
                                 const name_tensor_map& tensor_names,
                                 nvinfer1::ElementWiseOperation op_type) {
        const auto sources = param_get<picojson::array>(params, "sources");
        std::vector<nvinfer1::ITensor*> inputs;
        for(picojson::value source_element : sources) {
            const std::string& source = source_element.get<std::string>();
            auto source_tensor = tensor_names.find(source);
            if(source_tensor == tensor_names.end())
                return NULL;
            inputs.push_back(source_tensor->second);
        }
        assert(inputs.size() == 2);
        return network->addElementWise(*inputs[0], *inputs[1], op_type);
    }

    nvinfer1::ILayer* build_reshape(network_def network,
                                    const picojson::object& params,
                                    nvinfer1::DataType dt,
                                    const name_tensor_map& tensor_names) {
        (void)dt;

        const auto source = param_get<std::string>(params, "source");
        const auto shape = param_get<picojson::array>(params, "shape");
        nvinfer1::Dims dims = shapes_to_dims(shape);

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
        shuffle->setReshapeDimensions(dims);
        return shuffle;
    }

    nvinfer1::ILayer* build_transpose(network_def network,
                                      const picojson::object& params,
                                      const name_tensor_map& tensor_names) {
        const auto source = param_get<std::string>(params, "source");
        const auto axes = param_get<picojson::array>(params, "axes");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::Dims input_dim = input->getDimensions(),
                       output_dim = input_dim;

        nvinfer1::Permutation perm;
        for(size_t i = 0; i < axes.size(); ++i) {
            const int src_axis = (int)axes[i].get<double>();
            perm.order[i] = src_axis;
            output_dim.d[i] = input_dim.d[src_axis];
        }

        nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
        shuffle->setFirstTranspose(perm);
        shuffle->setReshapeDimensions(output_dim);
        return shuffle;
    }

    nvinfer1::ILayer*
    build_activation(network_def network, const picojson::object& params,
                     const name_tensor_map& tensor_names,
                     const nvinfer1::ActivationType activation_type) {
        const auto source = param_get<std::string>(params, "source");
        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        return network->addActivation(*input, activation_type);
    }

    nvinfer1::ILayer*
    build_pooling(network_def network, const picojson::object& params,
                  const name_tensor_map& tensor_names, nvinfer1::PoolingType pt,
                  std::shared_ptr<internal::output_dimensions_formula>
                    pool_outdim_formula) {
        const auto source = param_get<std::string>(params, "source");
        const auto name = param_get<std::string>(params, "name");
        const int window_w = param_get<int>(params, "window_width");
        const int window_h = param_get<int>(params, "window_height");
        const int stride_x = param_get<int>(params, "stride_x");
        const int stride_y = param_get<int>(params, "stride_y");
        const int pad_w = param_get<int>(params, "pad_w");
        const int pad_h = param_get<int>(params, "pad_h");
        const bool cover_all = param_get<bool>(params, "cover_all");

        pool_outdim_formula->cover_all_flags[name] = cover_all;

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::IPoolingLayer* pooling =
          network->addPooling(*input, pt, nvinfer1::DimsHW{window_h, window_w});
        pooling->setStride(nvinfer1::DimsHW{stride_y, stride_x});
        pooling->setPadding(nvinfer1::DimsHW{pad_h, pad_w});
        return pooling;
    }

    nvinfer1::ILayer*
    build_linear(network_def network, const picojson::object& params,
                 nvinfer1::DataType dt, const name_tensor_map& tensor_names,
                 const std::string& model_dir,
                 std::shared_ptr<internal::weights_manager> weights) {
        const auto source = param_get<std::string>(params, "source");
        const int n_out = param_get<int>(params, "n_out");
        const auto kernel_file =
          param_get<std::string>(params, "kernel_weights_file");
        const auto bias_file =
          param_get<std::string>(params, "bias_weights_file");

        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        nvinfer1::Weights kernel_weight =
          weights->load_weights_as(model_dir + "/" + kernel_file, dt);
        nvinfer1::Weights bias_weight =
          weights->load_weights_as(model_dir + "/" + bias_file, dt);
        return network->addFullyConnected(*input, n_out, kernel_weight,
                                          bias_weight);
    }

    nvinfer1::ILayer* build_softmax(network_def network,
                                    const picojson::object& params,
                                    const name_tensor_map& tensor_names) {
        const auto source = param_get<std::string>(params, "source");
        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::ITensor* input = source_tensor->second;
        return network->addSoftMax(*input);
    }

    nvinfer1::ILayer* build_concat(network_def network,
                                   const picojson::object& params,
                                   const name_tensor_map& tensor_names) {
        const int axis = params.find("axis") != params.end()
                           ? param_get<int>(params, "axis")
                           : 0;
        const auto source_elements =
          param_get<picojson::array>(params, "sources");
        nvinfer1::ITensor** sources =
          new nvinfer1::ITensor*[source_elements.size()];

        for(size_t i = 0; i < source_elements.size(); ++i) {
            const std::string& source = source_elements[i].get<std::string>();
            auto source_tensor = tensor_names.find(source);
            if(source_tensor == tensor_names.end())
                return NULL;
            sources[i] = source_tensor->second;
        }

        nvinfer1::IConcatenationLayer* concat =
          network->addConcatenation(sources, source_elements.size());
        concat->setAxis(axis);
        return concat;
    }

    nvinfer1::ILayer* build_unary(network_def network,
                                  const picojson::object& params,
                                  const name_tensor_map& tensor_names) {
        const auto type = param_get<std::string>(params, "type");
        const auto source = param_get<std::string>(params, "source");
        auto source_tensor = tensor_names.find(source);
        if(source_tensor == tensor_names.end())
            return NULL;

        nvinfer1::UnaryOperation op;
        if(type == "Unary")
            op = nvinfer1::UnaryOperation::kEXP;
        else
            assert(false);

        nvinfer1::ITensor* input = source_tensor->second;
        return network->addUnary(*input, op);
    }

    build_context
    make_network(std::shared_ptr<nvinfer1::IBuilder> builder,
                 const std::string& model_dir, nvinfer1::DataType dt,
                 std::shared_ptr<plugin::plugin_factory> factory, bool dla=false) {
        std::ifstream fs;
        fs.open((model_dir + "/model.json").c_str());

        assert(fs);

        picojson::value net;
        fs >> net;
        fs.close();

        const picojson::object& model = net.get<picojson::object>();
        const auto layers = param_get<picojson::array>(model, "layers");

        internal::build_context build_cxt;
        build_cxt.builder = builder;

        const auto inetwork_destroyer = [](nvinfer1::INetworkDefinition* n) {
            n->destroy();
        };
        build_cxt.network =
          network_def(builder->createNetwork(), inetwork_destroyer);

        // cover_all mode switcher (see also: chainer cover_all mode)
        build_cxt.pool_outdim_formula =
          std::make_shared<internal::output_dimensions_formula>();
        build_cxt.network->setPoolingOutputDimensionsFormula(
          build_cxt.pool_outdim_formula.get());

        name_tensor_map tensor_names;
        build_cxt.weights = std::make_shared<internal::weights_manager>();

        std::queue<picojson::value> layer_queue;
        for(size_t i = 0; i < layers.size(); i++)
            layer_queue.push(layers[i]);

        while(!layer_queue.empty()) {
            picojson::value layer = layer_queue.front();
            layer_queue.pop();
            picojson::object layer_params = layer.get<picojson::object>();

            const std::string& type = layer_params["type"].get<std::string>();
            const std::string& name = layer_params["name"].get<std::string>();
            if(get_verbose())
                std::cout << type << " - " << name << std::endl;

            // Check input
            if(type == "input") {
                internal::build_input(build_cxt.network, layer_params,
                                      tensor_names);
                continue;
            }

            nvinfer1::ILayer* l = NULL;

            // Check copy layer
            if(type == "Copy") {
                const std::string& source =
                  layer_params.find("source")->second.get<std::string>();
                auto source_tensor = tensor_names.find(source);
                if(source_tensor != tensor_names.end()) {
                    tensor_names[name] = source_tensor->second;
                    continue;
                }
            }

            // Check normal layers
            if(factory->is_registered(type))
                l = factory->build_plugin(build_cxt.network, layer_params, dt,
                                          tensor_names, model_dir);
            else if(type == "ConstantInput")
                l = build_constant_input(build_cxt.network, layer_params,
                                         model_dir, build_cxt.weights);
            else if(type == "Convolution2DFunction")
                l = build_conv(build_cxt.network, layer_params, dt,
                               tensor_names, model_dir, build_cxt.weights);
            else if(type == "Deconvolution2DFunction")
                l = build_deconv(build_cxt.network, layer_params, dt,
                                 tensor_names, model_dir, build_cxt.weights);
            else if(type == "BatchNormalizationFunction" ||
                    type == "BatchNormalization")
                l = build_bn(build_cxt.network, layer_params, dt, tensor_names,
                             model_dir, build_cxt.weights);
            else if(type == "ReLU")
                l = build_activation(build_cxt.network, layer_params,
                                     tensor_names,
                                     nvinfer1::ActivationType::kRELU);
            else if(type == "Sigmoid")
                l = build_activation(build_cxt.network, layer_params,
                                     tensor_names,
                                     nvinfer1::ActivationType::kSIGMOID);
            else if(type == "Tanh")
                l = build_activation(build_cxt.network, layer_params,
                                     tensor_names,
                                     nvinfer1::ActivationType::kTANH);
            else if(type == "MaxPooling2D")
                l = build_pooling(build_cxt.network, layer_params, tensor_names,
                                  nvinfer1::PoolingType::kMAX,
                                  build_cxt.pool_outdim_formula);
            else if(type == "AveragePooling2D")
                l = build_pooling(build_cxt.network, layer_params, tensor_names,
                                  nvinfer1::PoolingType::kAVERAGE,
                                  build_cxt.pool_outdim_formula);
            else if(type == "LinearFunction")
                l = build_linear(build_cxt.network, layer_params, dt,
                                 tensor_names, model_dir, build_cxt.weights);
            else if(type == "Softmax")
                l =
                  build_softmax(build_cxt.network, layer_params, tensor_names);
            else if(type == "LocalResponseNormalization")
                l = build_lrn(build_cxt.network, layer_params, tensor_names);
            else if(type == "MatMul")
                l = build_matmul(build_cxt.network, layer_params, tensor_names);
            else if(type == "Add")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kSUM);
            else if(type == "Sub")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kSUB);
            else if(type == "Mul")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kPROD);
            else if(type == "Div")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kDIV);
            else if(type == "Maximum")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kMAX);
            else if(type == "Minimum")
                l = build_eltw(build_cxt.network, layer_params, tensor_names,
                               nvinfer1::ElementWiseOperation::kMIN);
            else if(type == "Reshape")
                l = build_reshape(build_cxt.network, layer_params, dt,
                                  tensor_names);
            else if(type == "Transpose")
                l = build_transpose(build_cxt.network, layer_params,
                                    tensor_names);
            else if(type == "Concat")
                l = build_concat(build_cxt.network, layer_params, tensor_names);
            else if(type == "Unary")
                l = build_unary(build_cxt.network, layer_params, tensor_names);
            else if(type == "LinearInterpolate")
                l = build_linear_interpolate(build_cxt.network, layer_params,
                                             dt, tensor_names);
            else
                throw layer_not_implemented(name, type);

            if(!l) {
                // source layer is not built yet, so need to process afterwards
                std::cerr << type << " - " << name << std::endl;
                std::cerr << "source not found. will check later" << std::endl;
                layer_queue.push(layer);
                continue;
            }

            // Set naming
            l->setName(name.c_str());
            tensor_names[name] = l->getOutput(0);
            l->getOutput(0)->setName(name.c_str());

            // DLA support
            if(dla) {
                auto dla_flag = layer_params.find("dla");
                if(dla_flag != layer_params.end() &&
                    param_get<bool>(layer_params, "dla")) {
                    if(builder->canRunOnDLA(l)) {
                        builder->setDeviceType(l, nvinfer1::DeviceType::kDLA);
                    } else {
                        auto n = param_get<std::string>(layer_params, "name");
                        std::cerr << "DLA is specified for \"" << n;
                        std::cerr << "\" but TensorRT does not support.\n";
                    }
                }
            }
        }

        // Configure output layers
        // Originally, outputs are just an array of output layer names.
        // Currently, it's a list of list
        // [[layer_name1, output_name1],
        //  [layer_name2, output_name2],
        //  ...],
        // so that user can specify output name.
        // The original behavior is still supported for compatibility.
        const picojson::array& outputs =
          model.find("outputs")->second.get<picojson::array>();
        for(const picojson::value& output_layer_name : outputs) {
            std::string layer_name, output_name;
            if(output_layer_name.is<std::string>()) {
                // Unnamed output (use layer name as output name)
                output_name = layer_name = output_layer_name.get<std::string>();
            } else if(output_layer_name.is<picojson::array>()) {
                // Named output
                auto layer_name_and_output_name =
                  output_layer_name.get<picojson::array>();
                layer_name = layer_name_and_output_name[0].get<std::string>();
                output_name = layer_name_and_output_name[1].get<std::string>();
            } else {
                throw std::runtime_error("Unknown error in outputs field");
            }

            nvinfer1::ITensor* output_tensor = tensor_names[layer_name];
            output_tensor->setName(output_name.c_str());
            build_cxt.network->markOutput(*output_tensor);
        }
        return build_cxt;
    }

    std::shared_ptr<nvinfer1::IBuilder> make_builder(double workspace_gb,
                                                     int max_batch_size) {
        auto logger = logger::get_logger();
        auto p_builder = nvinfer1::createInferBuilder(*logger::get_logger());
        auto ibuilder_destroyer = [](nvinfer1::IBuilder* b) { b->destroy(); };
        auto builder =
          std::shared_ptr<nvinfer1::IBuilder>(p_builder, ibuilder_destroyer);
        builder->setMaxBatchSize(max_batch_size);
        builder->setMaxWorkspaceSize(
          size_t(workspace_gb * 1024L * 1024L * 1024L));
        builder->setDebugSync(get_verbose());
        return builder;
    }
}

void model::set_n_inputs_and_outputs() {
    const int nb_bindings = engine->getNbBindings();
    n_inputs = n_outputs = 0;
    for(int i = 0; i < nb_bindings; ++i)
        if(engine->bindingIsInput(i))
            n_inputs += 1;
        else
            n_outputs += 1;
}

std::shared_ptr<model> model::build(const build_param_fp32& param) {
    auto b = internal::make_builder(param.workspace_gb, param.max_batch_size);
    auto nw = internal::make_network(b, param.model_dir,
                                     nvinfer1::DataType::kFLOAT, param.factory);
    return nw.build();
}

std::shared_ptr<model> model::build(const build_param_fp16& param) {
    auto b = internal::make_builder(param.workspace_gb, param.max_batch_size);
    b->setHalf2Mode(true);
    b->allowGPUFallback(true);
    auto nw = internal::make_network(b, param.model_dir,
                                     nvinfer1::DataType::kHALF, param.factory,
                                     param.dla);
    return nw.build();
}

std::shared_ptr<model> model::build(const build_param_int8& param) {
    auto b = internal::make_builder(param.workspace_gb, param.max_batch_size);
    std::cout << "Int8 calibration enabled" << std::endl;
    b->setInt8Mode(true);

    auto build_cxt = internal::make_network(
      b, param.model_dir, nvinfer1::DataType::kFLOAT, param.factory);

    std::vector<nvinfer1::Dims> dims;
    for(int i = 0; i < build_cxt.network->getNbInputs(); ++i)
        dims.push_back(build_cxt.network->getInput(i)->getDimensions());

    auto calibrator = std::make_unique<internal::int8_entropy_calibrator>(
      dims, param.calib_stream, param.out_cache_file);
    b->setInt8Calibrator(calibrator.get());

    return build_cxt.build();
}

std::shared_ptr<model> model::build(const build_param_int8_cached& param) {
    auto b = internal::make_builder(param.workspace_gb, param.max_batch_size);
    std::cout << "Int8 calibration enabled (from cache \"";
    std::cout << param.in_cache_file << "\")" << std::endl;
    b->setInt8Mode(true);

    auto build_cxt = internal::make_network(
      b, param.model_dir, nvinfer1::DataType::kFLOAT, param.factory);

    auto calibrator =
      std::make_unique<internal::int8_entropy_calibrator_cached>(
        param.in_cache_file);
    b->setInt8Calibrator(calibrator.get());

    return build_cxt.build();
}

std::shared_ptr<model>
model::build_fp32(const std::string& model_dir, double workspace_gb,
                  int max_batch_size,
                  std::shared_ptr<plugin::plugin_factory> factory) {
    build_param_fp32 p(model_dir, workspace_gb, max_batch_size, factory);
    return build(p);
}

std::shared_ptr<model>
model::build_fp16(const std::string& model_dir, double workspace_gb,
                  int max_batch_size,
                  std::shared_ptr<plugin::plugin_factory> factory, bool dla) {
    build_param_fp16 p(model_dir, workspace_gb, max_batch_size, factory);
    p.dla = dla;
    return build(p);
}

std::shared_ptr<model>
model::build_int8(const std::string& model_dir,
                  std::shared_ptr<calibration_stream> calib_stream,
                  double workspace_gb, int max_batch_size,
                  const std::string& out_cache_file,
                  std::shared_ptr<plugin::plugin_factory> factory) {
    build_param_int8 p(model_dir, calib_stream, workspace_gb, max_batch_size,
                       out_cache_file, factory);
    return build(p);
}

std::shared_ptr<model>
model::build_int8_cache(const std::string& model_dir,
                        const std::string& in_cache_file, double workspace_gb,
                        int max_batch_size,
                        std::shared_ptr<plugin::plugin_factory> factory) {
    if(!in_cache_file.size())
        std::invalid_argument("in_cache_file shouldn't be empty");
    build_param_int8_cached p(model_dir, in_cache_file, workspace_gb,
                              max_batch_size, factory);
    return build(p);
}

std::shared_ptr<model>
model::deserialize(std::istream& ist,
                   std::shared_ptr<plugin::plugin_factory> factory) {
    // Calculate buffer size for the actual tensorrt serialized data
    const auto current_pos_in_stream = ist.tellg();
    ist.seekg(0, std::ios::end);
    const auto size = ist.tellg() - current_pos_in_stream;

    // Load serialized tensorrt engine
    ist.seekg(current_pos_in_stream, std::ios::beg);
    std::vector<char> buf(size);
    ist.read(buf.data(), size);

    // Deserialize
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto rt_model = std::shared_ptr<model>(new model());
    auto runtime = internal::unique_ptr_with_destroyer<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(*logger::get_logger()));
    auto p_engine =
      runtime->deserializeCudaEngine((void*)buf.data(), size, factory.get());
    rt_model->engine =
      internal::unique_ptr_with_destroyer<nvinfer1::ICudaEngine>(p_engine);
    rt_model->set_n_inputs_and_outputs();
    return rt_model;
}

std::shared_ptr<model>
model::deserialize(const std::string& model_file,
                   std::shared_ptr<plugin::plugin_factory> factory) {
    std::ifstream ifs(model_file, std::ios::in | std::ios::binary);
    return model::deserialize(ifs, factory);
}

void model::serialize(std::ostream& ost) const {
    auto model_stream =
      internal::unique_ptr_with_destroyer<nvinfer1::IHostMemory>(
        engine->serialize());
    ost.write((char*)model_stream->data(), model_stream->size());
}

void model::serialize(const std::string& model_file) const {
    std::ofstream ofs(model_file, std::ios::out | std::ios::binary);
    model::serialize(ofs);
}

int model::get_max_batch_size() const {
    return engine->getMaxBatchSize();
}

int model::get_n_inputs() const {
    return n_inputs;
}

int model::get_n_outputs() const {
    return n_outputs;
}

std::vector<std::string> model::get_input_names() const {
    std::vector<std::string> names;
    for(int i = 0; i < get_n_inputs(); ++i)
        names.push_back(engine->getBindingName(i));
    return names;
}

std::vector<std::string> model::get_output_names() const {
    std::vector<std::string> names;
    for(int i = 0; i < get_n_outputs(); ++i)
        names.push_back(engine->getBindingName(get_n_inputs() + i));
    return names;
}

std::vector<int> model::get_binding_dimensions(int index) const {
    const nvinfer1::Dims dims = engine->getBindingDimensions(index);
    return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

std::vector<int> model::get_input_dimensions(int input_index) const {
    if(input_index < 0 || n_inputs <= input_index)
        throw std::invalid_argument("input_index out of range");
    return get_binding_dimensions(input_index);
}

std::vector<int> model::get_input_dimensions(const std::string& name) const {
    if(!has_input(name))
        throw std::invalid_argument(
          "the network doesn't have the specified input name " + name);
    return get_binding_dimensions(engine->getBindingIndex(name.c_str()));
}

std::vector<int> model::get_output_dimensions(int output_index) const {
    if(output_index < 0 || n_outputs <= output_index)
        throw std::invalid_argument("output_index out of range");
    return get_binding_dimensions(n_inputs + output_index);
}

std::vector<int> model::get_output_dimensions(const std::string& name) const {
    if(!has_output(name))
        throw std::invalid_argument(
          "the network doesn't have the specified output name " + name);
    return get_binding_dimensions(engine->getBindingIndex(name.c_str()));
}

bool model::has_input(const std::string& input_name) const {
    // getBindingIndex returns -1 if the name doesn't exist
    const int bind_idx = engine->getBindingIndex(input_name.c_str());
    return 0 <= bind_idx && bind_idx < get_n_inputs();
}

bool model::has_output(const std::string& output_name) const {
    // getBindingIndex returns -1 if the name doesn't exist
    const int bind_idx = engine->getBindingIndex(output_name.c_str());
    // binding index contains number of user inputs, so subtract
    const int output_idx = bind_idx - get_n_inputs();
    return 0 <= output_idx && output_idx < get_n_outputs();
}
}
