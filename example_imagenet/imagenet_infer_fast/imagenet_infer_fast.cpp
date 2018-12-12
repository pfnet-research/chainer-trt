/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "chainer_trt/chainer_trt.hpp"
#include "chainer_trt/utils.hpp"
#include "cmdline.h"

// TODO: need to fix the situation where the number of total data size is
// not divisible by batch size (make_batch)

using file_label_pair = std::tuple<std::string, int>;
using batch_t = std::vector<file_label_pair>;

static std::vector<file_label_pair> load_images(const std::string& filename,
                                                const std::string& base_path) {
    std::vector<file_label_pair> images;
    std::ifstream ifs(filename);
    while(!ifs.eof()) {
        std::string file;
        int label;
        ifs >> file >> label;
        if(file.size() == 0)
            break;
        images.push_back(std::make_tuple(base_path + "/" + file, label));
    }
    return images;
}

static std::queue<std::tuple<int, batch_t>>
make_batch_queue(const std::vector<file_label_pair>& images, unsigned n_batch) {
    std::queue<std::tuple<int, batch_t>> result;
    unsigned i_batch = 0;
    for(unsigned i = 0; i < images.size();) {
        batch_t b;
        for(unsigned j = 0; j < n_batch && i < images.size(); ++j, ++i)
            b.push_back(images[i]);
        result.push(std::make_tuple(i_batch++, b));
    }
    return result;
}

static void make_batch(void* ptr, const batch_t& b, int img_w, int img_h) {
    std::vector<float> buf(img_w * img_h * 3 * b.size());
    const int elements_per_image = img_w * img_h * 3;
    for(unsigned i = 0; i < b.size(); ++i) {
        auto img = cv::imread(std::get<0>(b[i]));
        cv::resize(img, img, cv::Size(img_w, img_h));
        cv::Mat image_fp32;
        img.convertTo(image_fp32, CV_32F);
        const float* p = (float*)image_fp32.data;
        std::copy(p, p + elements_per_image,
                  buf.data() + i * elements_per_image);
    }
    std::copy(buf.begin(), buf.end(), (float*)ptr);
}

class infer_worker {
    int batch_size;
    int img_w, img_h;
    std::unique_ptr<chainer_trt::infer> engine;
    std::shared_ptr<chainer_trt::buffer> buf;
    std::queue<std::tuple<int, batch_t>>& ref_queue;
    std::mutex& ref_mtx;

    std::unique_ptr<unsigned char[]> batch_in;
    float* all_out;

public:
    infer_worker(std::shared_ptr<chainer_trt::model>& model, int _batch_size,
                 int _img_w, int _img_h,
                 std::queue<std::tuple<int, batch_t>>& q, std::mutex& _mtx,
                 float* _all_out)
      : batch_size(_batch_size), img_w(_img_w), img_h(_img_h),
        engine(new chainer_trt::infer(model)),
        buf(engine->create_buffer(_batch_size)), ref_queue(q), ref_mtx(_mtx),
        batch_in(
          new unsigned char[img_w * img_h * 3 * _batch_size * sizeof(float)]),
        all_out(_all_out) {}

    bool queue_empty() const {
        std::lock_guard<std::mutex> lock(ref_mtx);
        return ref_queue.empty();
    }

    std::tuple<int, batch_t> get_next() {
        std::lock_guard<std::mutex> lock(ref_mtx);
        std::tuple<int, batch_t> b = ref_queue.front();
        ref_queue.pop();
        return b;
    }

    void operator()() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        while(!queue_empty()) {
            // Load batch
            std::tuple<int, batch_t> b = get_next();
            const int out_offset = std::get<0>(b) * batch_size * 1000;

            make_batch(batch_in.get(), std::get<1>(b), img_w, img_h);

            // Wait for the buffer being available
            cudaStreamSynchronize(stream);

            auto in = std::vector<const void*>{(const void*)batch_in.get()};
            auto out = std::vector<void*>{(void*)(all_out + out_offset)};
            buf->input_host_to_device(in, stream);
            (*engine)(*buf, stream);
            buf->output_device_to_host(out, stream);
        }
    }
};

static std::vector<std::tuple<float, int>> sort_output(const float* output,
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

    const auto images = load_images(args.images_arg, args.base_path_arg);
    auto queue = make_batch_queue(images, args.batch_arg);

    std::mutex mutex;
    float* all_out = new float[1000 * images.size()];
    std::fill(all_out, all_out + 1000 * images.size(), 0.0);

    std::cout << "Running inference" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    auto model = chainer_trt::model::deserialize(args.model_arg);
    for(int i = 0; i < args.n_stream_arg; ++i)
        workers.push_back(std::thread(infer_worker(
          model, args.batch_arg, 224, 224, queue, mutex, all_out)));

    for(std::thread& worker : workers)
        worker.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    const int total_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const double image_us = total_us / images.size();
    const double batch_us = image_us * args.batch_arg;

    // Evaluation
    int top1_n_tp = 0, top5_n_tp = 0;
    for(unsigned i = 0; i < images.size(); ++i) {
        const int ofst = i * 1000;
        auto score_and_index = sort_output(all_out + ofst, 1000);
        if(std::get<1>(score_and_index[0]) == std::get<1>(images[i]))
            top1_n_tp++;

        for(int j = 0; j < 5; ++j) {
            if(std::get<1>(score_and_index[j]) == std::get<1>(images[i])) {
                top5_n_tp++;
                break;
            }
        }
    }

    const float top1_acc = (100.0 * top1_n_tp / images.size());
    const float top5_acc = (100.0 * top5_n_tp / images.size());
    std::cout << "top1 accuracy " << top1_acc << "%" << std::endl;
    std::cout << "top5 accuracy " << top5_acc << "%" << std::endl;
    std::cout << "total time " << (total_us / 1e6) << "s" << std::endl;
    std::cout << "average time " << (batch_us / 1000) << "ms/batch ";
    std::cout << "(" << (image_us / 1000) << "ms/image)" << std::endl;
}
