# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.


class ModuleNotYetBuiltError(Exception):
    def __init__(self):
        msg = "chainer_trt.Buffer and chainer_trt." \
              "Infer cannot be initialized because chainer-trt " \
              "Python module is not yet built.\n" \
              "Please build chainer-trt with -DWITH_PYTHON_LIB=yes " \
              "cmake option, and put libpyrt.so in a correct place where " \
              "the Python interpreter can find."
        super(ModuleNotYetBuiltError, self).__init__(msg)


class Buffer(object):
    def __init__(self, _, __):
        raise ModuleNotYetBuiltError()


class Infer(object):
    def __init__(self, _):
        raise ModuleNotYetBuiltError()
