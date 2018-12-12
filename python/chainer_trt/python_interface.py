# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import chainer
import chainer.cuda
import cupy
import numpy

import libpyrt


class Buffer(object):
    def __init__(self, infer, batch_size):
        self.batch_size = batch_size
        self.native_buffer = infer.native_infer.create_buffer(batch_size)
        out_shapes = infer.native_infer.get_output_shapes()
        self.output_shapes = [tuple(s) for s in out_shapes]
        self.output_names = infer.native_infer.get_output_names()
        self.debug = infer.debug

    def input_to_gpu(self, inputs_np):
        # TODO(suzuo): shape check if debug flag is on
        if type(inputs_np) == list:
            xs = [x if x.flags['C_CONTIGUOUS'] else numpy.ascontiguousarray(x)
                  for x in inputs_np]
            x_ptrs = [x.ctypes.data for x in xs]
            self.native_buffer.input_host_to_device(x_ptrs)
        elif type(inputs_np) == dict:
            xs = {name: x if x.flags['C_CONTIGUOUS']
                  else numpy.ascontiguousarray(x)
                  for name, x in inputs_np.items()}
            x_ptrs = {name: x.ctypes.data for name, x in xs.items()}
            self.native_buffer.input_host_to_device_with_name(x_ptrs)

    def output_to_cpu(self, outputs_np):
        # TODO(suzuo): shape check if debug flag is on
        if type(outputs_np) == list:
            y_ptrs = [y.ctypes.data for y in outputs_np]
            self.native_buffer.output_device_to_host(y_ptrs)
        elif type(outputs_np) == dict:
            y_ptrs = {name: y.ctypes.data for name, y in outputs_np.items()}
            self.native_buffer.output_device_to_host_with_name(y_ptrs)


class Infer(object):
    def __init__(self, model_file):
        if type(model_file) == str:
            self.native_infer = libpyrt.Infer(model_file)
        elif type(model_file) == libpyrt.Infer:
            self.native_infer = model_file
        in_shapes = self.native_infer.get_input_shapes()
        out_shapes = self.native_infer.get_output_shapes()
        self.input_shapes = [tuple(s) for s in in_shapes]
        self.output_shapes = [tuple(s) for s in out_shapes]
        self.output_names = self.native_infer.get_output_names()
        self.debug = False

    @classmethod
    def build(cls, export_file):
        """Should be used from internal test purpose only"""
        return Infer(libpyrt.Infer.build(export_file))

    def create_buffer(self, batch_size):
        # type: (int) -> Buffer
        return Buffer(self, batch_size)

    def __get_batch_size(self, input_values):
        s = input_values[0].shape
        if len(s) == len(self.input_shapes[0]):
            # x=(3, 224, 224), input_shapes[0]=(3, 224, 224)
            #   -> bs=1
            return 1
        elif len(s) == len(self.input_shapes[0]) + 1:
            # x=(8, 3, 224, 224), input_shapes[0]=(3, 224, 224)
            #   -> bs=x.shape[0]=8
            return s[0]
        else:
            raise ValueError("Input shape mismatch")

    def __run_buffer(self, inputs):
        if self.debug:
            if 1 < len(inputs):
                msg = "When specifying Buffer to Infer.__call__, " \
                      "nothing else should be given"
                raise ValueError(msg)
        self.native_infer(inputs[0].native_buffer)

    def __run_named_array(self, inputs):
        input_values = list(inputs.values())
        xp = chainer.cuda.get_array_module(input_values[0])
        batch_size = self.__get_batch_size(input_values)

        # Make inputs contiguous
        xs = {name: x if x.flags['C_CONTIGUOUS']
              else xp.ascontiguousarray(x)
              for name, x in inputs.items()}

        # Make outputs with names
        ys = {name: xp.ndarray((batch_size,) + s, dtype=xp.float32)
              for name, s in zip(self.output_names, self.output_shapes)}

        if xp == cupy:
            x_ptrs = {name: x.data.ptr for name, x in xs.items()}
            y_ptrs = {name: y.data.ptr for name, y in ys.items()}
            self.native_infer.infer_from_gpu_with_name(batch_size, x_ptrs,
                                                       y_ptrs)
        else:
            x_ptrs = {name: x.ctypes.data for name, x in xs.items()}
            y_ptrs = {name: y.ctypes.data for name, y in ys.items()}
            self.native_infer.infer_from_cpu_with_name(batch_size, x_ptrs,
                                                       y_ptrs)
        return ys

    def __run_list_of_arrays(self, inputs):
        inputs = inputs[0]
        if self.debug:
            # TODO(suzuo): shape check
            all_np = all(type(x) is numpy.ndarray for x in inputs)
            all_cp = all(type(x) is cupy.ndarray for x in inputs)
            if not all_np and not all_cp:
                msg = "When specifying a list to Infer.__call__, " \
                    "it should be all numpy.ndarray or all cupy.ndarray"
                raise ValueError(msg)

        xp = chainer.cuda.get_array_module(inputs[0])
        batch_size = self.__get_batch_size(inputs)
        xs = [x if x.flags['C_CONTIGUOUS'] else xp.ascontiguousarray(x)
              for x in inputs]
        ys = [xp.ndarray((batch_size,) + s, dtype=xp.float32)
              for s in self.output_shapes]

        if xp == cupy:
            x_ptrs = [x.data.ptr for x in xs]
            y_ptrs = [y.data.ptr for y in ys]
            self.native_infer.infer_from_gpu(batch_size, x_ptrs, y_ptrs)
        else:
            x_ptrs = [x.ctypes.data for x in xs]
            y_ptrs = [y.ctypes.data for y in ys]
            self.native_infer.infer_from_cpu(batch_size, x_ptrs, y_ptrs)
        return ys

    def __call__(self, *inputs, **kwargs):
        """Run inference

        :param inputs:
            * Buffer
            * list of numpy.array or cupy.array
            * varargs of numpy.array or cupy.array
        """
        if len(kwargs) != 0:
            if len(inputs) != 0:
                raise ValueError("")
            # case: infer(x1=x1, x2=x2)
            return self.__run_named_array(kwargs)
        if len(inputs) == 0:
            raise ValueError("Nothing is specified to __call__")

        input_type = type(inputs[0])
        if input_type is Buffer:
            # case: infer(buf)
            self.__run_buffer(inputs)
        elif input_type is dict:
            # case: infer({'input': x})
            return self.__run_named_array(inputs[0])
        elif input_type is list:
            # case: infer([x1, x2])
            return self.__run_list_of_arrays(inputs)
        elif 1 < len(inputs) or input_type in (numpy.ndarray, cupy.ndarray):
            # case: infer(x), infer(x1, x2)
            return self([*inputs])
        else:
            raise ValueError("Unknown argument type.")
