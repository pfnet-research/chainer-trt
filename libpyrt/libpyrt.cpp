/*
 * Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop

#include <glog/logging.h>

#include <chainer_trt/chainer_trt.hpp>

namespace py = pybind11;

namespace chainer_trt {
namespace internal {

    template <typename TP>
    std::map<std::string, TP>
    convert_ptrs(const std::map<std::string, uintptr_t>& ptrs) {
        std::map<std::string, TP> bufs;
        for(auto it : ptrs)
            bufs[it.first] = (TP)it.second;
        return bufs;
    }

    template <typename TP>
    std::vector<TP> convert_ptrs(const std::vector<uintptr_t>& ptrs) {
        std::vector<TP> bufs;
        for(uintptr_t p : ptrs)
            bufs.push_back((TP)p);
        return bufs;
    }

    class infer_py;

    class buffer_py {
        friend class infer_py;

        std::shared_ptr<chainer_trt::buffer> buf;
        buffer_py() = delete;

        buffer_py(std::shared_ptr<chainer_trt::buffer> _buf) : buf(_buf) {}

    public:
        void input_host_to_device(const std::vector<uintptr_t>& inputs_cpu) {
            buf->input_host_to_device(convert_ptrs<const void*>(inputs_cpu));
        }

        void input_host_to_device_with_name(
          const std::map<std::string, uintptr_t>& inputs_cpu) {
            buf->input_host_to_device(convert_ptrs<const void*>(inputs_cpu));
        }

        void
        output_device_to_host(const std::vector<uintptr_t>& outputs_cpu) const {
            buf->output_device_to_host(convert_ptrs<void*>(outputs_cpu));
        }

        void output_device_to_host_with_name(
          const std::map<std::string, uintptr_t>& outputs_cpu) const {
            buf->output_device_to_host(convert_ptrs<void*>(outputs_cpu));
        }

        int get_batch_size() const { return buf->get_batch_size(); }
    };

    class infer_py {
        std::shared_ptr<chainer_trt::model> m;
        std::shared_ptr<chainer_trt::infer> rt;

        infer_py(std::shared_ptr<chainer_trt::model> _m)
          : m(_m), rt(std::make_shared<chainer_trt::infer>(m)) {}

    public:
        infer_py(const std::string& model_path)
          : infer_py(chainer_trt::model::deserialize(model_path)) {}

        // This is used for testing. Please don't use it in user code.
        static infer_py build_infer_py(const std::string& export_path) {
            return infer_py(chainer_trt::model::build_fp32(export_path));
        }

        std::vector<std::vector<int>> get_input_shapes() const {
            std::vector<std::vector<int>> shapes;
            for(int i = 0; i < m->get_n_inputs(); ++i)
                shapes.push_back(m->get_input_dimensions(i));
            return shapes;
        }

        std::vector<std::vector<int>> get_output_shapes() const {
            std::vector<std::vector<int>> shapes;
            for(int i = 0; i < m->get_n_outputs(); ++i)
                shapes.push_back(m->get_output_dimensions(i));
            return shapes;
        }

        std::vector<std::string> get_output_names() const {
            return m->get_output_names();
        }

        void infer_from_cpu(int batch_size,
                            const std::vector<uintptr_t>& in_ptrs,
                            const std::vector<uintptr_t>& out_ptrs) {
            rt->infer_from_cpu(batch_size, convert_ptrs<const void*>(in_ptrs),
                               convert_ptrs<void*>(out_ptrs));
        }

        void infer_from_cpu_with_name(
          int batch_size, const std::map<std::string, uintptr_t>& in_ptrs,
          const std::map<std::string, uintptr_t>& out_ptrs) {
            rt->infer_from_cpu(batch_size, convert_ptrs<const void*>(in_ptrs),
                               convert_ptrs<void*>(out_ptrs));
        }

        void infer_from_gpu(int batch_size,
                            const std::vector<uintptr_t>& in_ptrs_gpu,
                            const std::vector<uintptr_t>& out_ptrs_gpu) {
            (*rt)(batch_size, convert_ptrs<const void*>(in_ptrs_gpu),
                  convert_ptrs<void*>(out_ptrs_gpu));
        }

        void infer_from_gpu_with_name(
          int batch_size, const std::map<std::string, uintptr_t>& in_ptrs_gpu,
          const std::map<std::string, uintptr_t>& out_ptrs_gpu) {
            (*rt)(batch_size, convert_ptrs<const void*>(in_ptrs_gpu),
                  convert_ptrs<void*>(out_ptrs_gpu));
        }

        buffer_py create_buffer(int batch_size) const {
            return buffer_py(rt->create_buffer(batch_size));
        }

        void operator()(buffer_py& buf) { (*rt)(*buf.buf); }
    };
}
}

using infer_py = chainer_trt::internal::infer_py;
using buffer_py = chainer_trt::internal::buffer_py;

PYBIND11_MODULE(libpyrt, m) {
    google::InitGoogleLogging("libpyrt");
    google::InstallFailureSignalHandler();

    m.doc() = "Python inferface to run TensorRT inference engine";

    py::class_<infer_py> infer(m, "Infer");
    infer.def(py::init<const std::string&>())
      .def("build", &infer_py::build_infer_py)
      .def("create_buffer", &infer_py::create_buffer)
      .def("get_input_shapes", &infer_py::get_input_shapes)
      .def("get_output_shapes", &infer_py::get_output_shapes)
      .def("get_output_names", &infer_py::get_output_names)
      .def("infer_from_cpu", &infer_py::infer_from_cpu)
      .def("infer_from_cpu_with_name", &infer_py::infer_from_cpu_with_name)
      .def("infer_from_gpu", &infer_py::infer_from_gpu)
      .def("infer_from_gpu_with_name", &infer_py::infer_from_gpu_with_name)
      .def("__call__", &infer_py::operator());

    py::class_<buffer_py> buffer(m, "Buffer");
    buffer.def("input_host_to_device", &buffer_py::input_host_to_device)
      .def("input_host_to_device_with_name",
           &buffer_py::input_host_to_device_with_name)
      .def("get_batch_size", &buffer_py::get_batch_size)
      .def("output_device_to_host", &buffer_py::output_device_to_host)
      .def("output_device_to_host_with_name",
           &buffer_py::output_device_to_host_with_name);
}
