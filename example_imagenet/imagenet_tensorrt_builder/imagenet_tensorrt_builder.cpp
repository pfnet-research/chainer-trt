/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include <glog/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/utils.hpp"
#include "cmdline.h"

class imagenet_batch_stream : public chainer_trt::calibration_stream {
    std::vector<std::string> image_files;

public:
    imagenet_batch_stream(const std::string& list_file) {
        std::ifstream ifs(list_file);
        while(!ifs.eof()) {
            std::string t;
            std::getline(ifs, t);
            if(t.size() == 0)
                break;
            image_files.push_back(t);
        }
    }

    virtual int get_n_batch() override { return image_files.size(); }
    virtual int get_n_input() override { return 1; }

    virtual void get_batch(int i_batch, int input_idx,
                           const std::vector<int>& dims,
                           void* dst_buf_cpu) override {
        (void)input_idx;

        assert(dims.size() == 3); // assumes hwc
        assert(dims[2] == 3);     // channel dim

        cv::Mat image = cv::imread(image_files[i_batch]);
        if(image.data == NULL)
            throw "Image not found";
        if(image.rows != dims[0] || image.cols != dims[1])
            cv::resize(image, image, cv::Size(dims[0], dims[1]));

        cv::Mat img_float;
        image.convertTo(img_float, CV_32F);
        const float* data_p = (float*)img_float.data;
        const int data_size = dims[0] * dims[1] * dims[2];
        std::copy(data_p, data_p + data_size, (float*)dst_buf_cpu);
    }
};

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    gengetopt_args_info args;
    if(cmdline_parser(argc, argv, &args))
        return -1;

    std::string msg = "";
    if(args.mode_arg == std::string("int8")) {
        if(args.in_cache_given && args.out_cache_given)
            msg = "--in-cache and --out-cache cannot be specified together";
        else if(args.in_cache_given && args.calib_given)
            msg = "--calib and --in-cache cannot be specified together";
        else if(!args.in_cache_given && !args.calib_given)
            msg =
              "either --calib or --in-cache has to be specified "
              "when using \"--mode int8\"";
    } else {
        if(args.calib_given)
            msg += "--calib ";
        if(args.in_cache_given)
            msg += "--in-cache ";
        if(args.out_cache_given)
            msg += "--out-cache ";

        if(!msg.empty())
            msg += "option(s) need to be specified with \"--mode int8\"";
    }
    if(args.dla_flag && args.mode_arg != std::string("fp16"))
        msg += "--dla has to be specified together with --mode fp16";
    if(!msg.empty()) {
        std::cerr << msg << std::endl << std::endl;
        cmdline_parser_print_help();
        return -1;
    }

    if(args.verbose_given) {
        std::cerr << "Verbose mode" << std::endl;
        chainer_trt::set_verbose(true);
    }

    cudaSetDevice(args.gpu_arg);
    show_device_info();

    std::shared_ptr<chainer_trt::model> m;
    if(args.mode_arg == std::string("int8") && !args.in_cache_given) {
        auto calib_stream =
          std::make_shared<imagenet_batch_stream>(args.calib_arg);

        m = chainer_trt::model::build_int8(
          args.dir_arg, calib_stream, args.workspace_arg, args.max_batch_arg,
          args.out_cache_arg);
    } else if(args.mode_arg == std::string("int8") && args.in_cache_given) {
        m = chainer_trt::model::build_int8_cache(
          args.dir_arg, args.in_cache_arg, args.workspace_arg,
          args.max_batch_arg);
    } else if(args.mode_arg == std::string("fp16")) {
        auto p = chainer_trt::build_param_fp16(args.dir_arg, args.workspace_arg,
                                               args.max_batch_arg);
        p.dla = args.dla_flag;
        m = chainer_trt::model::build(p);
    } else {
        m = chainer_trt::model::build_fp32(args.dir_arg, args.workspace_arg,
                                           args.max_batch_arg);
    }

    m->serialize(args.model_arg);
}
