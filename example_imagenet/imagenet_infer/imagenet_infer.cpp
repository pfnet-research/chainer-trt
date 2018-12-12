/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>

#include <glog/logging.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/profiling.hpp"
#include "chainer_trt/utils.hpp"
#include "cmdline.h"

using namespace std::literals::string_literals;

std::vector<std::string> load_label_file(const char* label_file) {
    std::vector<std::string> labels;
    std::ifstream ifs(label_file);
    if(!ifs)
        throw std::string(label_file) + " couldn't be opened.";
    for(std::string line; std::getline(ifs, line);)
        labels.push_back(line);
    return labels;
}

std::vector<std::tuple<float, int>> sort_output(const float* output,
                                                const int len) {
    std::vector<std::tuple<float, int>> scores;
    for(int i = 0; i < len; ++i)
        scores.push_back(std::make_tuple(output[i], i));
    std::sort(scores.rbegin(), scores.rend());
    return scores;
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    gengetopt_args_info args;
    if(cmdline_parser(argc, argv, &args))
        return -1;

    cudaSetDevice(args.gpu_arg);
    show_device_info();

    std::cout << "Batch-size = " << args.batch_size_arg << std::endl;

    std::cout << "Loading model" << std::endl;
    std::shared_ptr<chainer_trt::model> m =
      chainer_trt::model::deserialize(args.model_arg);

    std::shared_ptr<chainer_trt::default_profiler> prof;
    if(args.prof_given)
        prof = std::make_shared<chainer_trt::default_profiler>();

    chainer_trt::infer rt(m, prof);

    std::cout << "Loading labels" << std::endl;
    const std::vector<std::string> labels = load_label_file(args.label_arg);

    std::cout << "Loading image" << std::endl;
    const int img_size = 224;
    cv::Mat image = cv::imread(args.image_arg);
    if(image.rows != img_size || image.cols != img_size)
        cv::resize(image, image, cv::Size(img_size, img_size));
    cv::Mat image_fp32;
    image.convertTo(image_fp32, CV_32F);

    const int n_vals_per_img = img_size * img_size * 3;
    float* input = new float[n_vals_per_img * args.batch_size_arg];
    float* output = new float[1000 * args.batch_size_arg];

    // Copy first image to the entire batch
    for(int batch = 0; batch < args.batch_size_arg; ++batch)
        std::copy((float*)image_fp32.data,
                  (float*)image_fp32.data + n_vals_per_img,
                  input + batch * n_vals_per_img);

    auto buf = rt.create_buffer(args.batch_size_arg);

    std::cout << "Send input to GPU" << std::endl;
    buf->input_host_to_device({{"input"s, input}});

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < args.n_try_arg; ++i)
        rt(*buf);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    double avg_us = us.count() / args.n_try_arg;
    std::cout << "Average inference time = ";
    std::cout << (avg_us / 1000) << "ms" << std::endl;

    std::cout << "Get output from GPU" << std::endl;
    buf->output_device_to_host({{"prob"s, output}});

    auto score_and_index = sort_output(output, 1000);
    for(int i = 0; i < 5; ++i) {
        const std::string& label = labels[std::get<1>(score_and_index[i])];
        const float score = std::get<0>(score_and_index[i]);
        std::cout << std::fixed << std::setprecision(6) << score;
        std::cout << " - " << label << std::endl;
    }

    // Verify another result in the batch
    for(int batch = 1; batch < args.batch_size_arg; ++batch) {
        auto score_and_index_batch = sort_output(output + 1000 * batch, 1000);
        bool corrupted = false;
        for(int i = 0; i < 5; ++i)
            corrupted |= (std::get<0>(score_and_index[i]) !=
                          std::get<0>(score_and_index_batch[i]));

        if(corrupted) {
            std::cout << "batch " << batch << " is corrupted" << std::endl;
            for(int i = 0; i < 5; ++i) {
                const int index = std::get<1>(score_and_index_batch[i]);
                const std::string& label = labels[index];
                const float score = std::get<0>(score_and_index_batch[i]);
                std::cout << std::fixed << std::setprecision(6) << score;
                std::cout << " - " << label << std::endl;
            }
        }
    }

    if(prof)
        prof->show_profiling_result(std::cout, args.prof_arg);
}
