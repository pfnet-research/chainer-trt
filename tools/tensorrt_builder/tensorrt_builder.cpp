/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/utils.hpp"
#include "cmdline.h"

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    gengetopt_args_info args;
    if(cmdline_parser(argc, argv, &args))
        return -1;

    cudaSetDevice(args.gpu_arg);
    show_device_info();

    std::shared_ptr<chainer_trt::model> m;
    if(args.mode_arg == std::string("fp16"))
        m = chainer_trt::model::build_fp16(args.dir_arg, args.workspace_arg,
                                           args.max_batch_arg);
    else
        m = chainer_trt::model::build_fp32(args.dir_arg, args.workspace_arg,
                                           args.max_batch_arg);

    m->serialize(args.model_arg);
}
