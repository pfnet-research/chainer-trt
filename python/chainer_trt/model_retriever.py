# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import json
import os
import traceback
import warnings

from collections import OrderedDict
from distutils.version import StrictVersion

import numpy as np

import chainer
from chainer.computational_graph import build_computational_graph
import chainer.functions
import chainer.functions as F
from chainer.functions.math.basic_math import Add
from chainer.functions.math.basic_math import Div
from chainer.functions.math.basic_math import Mul
from chainer.functions.math.basic_math import Sub

from chainer.functions.math.basic_math import AddConstant
from chainer.functions.math.basic_math import DivFromConstant
from chainer.functions.math.basic_math import MulConstant
from chainer.functions.math.basic_math import SubFromConstant

from chainer.functions.connection.convolution_2d \
    import Convolution2DFunction
from chainer.functions.connection.deconvolution_2d \
    import Deconvolution2DFunction

import chainer_trt.functions
from chainer_trt.json_encoder import JSONEncoderEX


def chainer_ver():
    return StrictVersion(chainer.__version__)


class RetainHook(chainer.FunctionHook):
    def forward_postprocess(self, function, in_data):
        function.retain_inputs(list(range(len(in_data))))


class MarkPrefixHook(chainer.FunctionHook):
    """Mark a prefix to the all the chainer functions executed during the hook lifetime.

    The marked prefix will appear in layer information exported in JSON
    """     # NOQA
    def __init__(self, prefix):
        self._layer_prefix = prefix
        self.name = 'MarkHook_' + prefix

    def forward_postprocess(self, function, _):
        if hasattr(function, '_MarkPrefixHook__chainer_trt_prefix'):
            pfx = self._layer_prefix + '-' + function.__chainer_trt_prefix
        else:
            pfx = self._layer_prefix
        function.__chainer_trt_prefix = pfx


class TracebackHook(chainer.FunctionHook):
    """Save Traceback to exported network"""
    name = 'TracebackHook'

    def forward_postprocess(self, function, _):
        tb = traceback.extract_stack()[:-2]
        tb = traceback.format_list(tb)
        function.__chainer_trt_traceback = ''.join(tb).strip()


class ModelRetriever(object):
    RANK_CONSTANT_INPUT = -1
    RANK_INPUT = -2

    def __init__(self, dst_path, verbose=False):
        self.verbose = verbose

        self.dst_path = dst_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # user by register_inputs, key:variable and val:string
        # (make variables as user inputs (and record their name))
        self.registered_user_input_names = OrderedDict()

        self.layers = []
        self.output_variables = []
        self.output_names = OrderedDict()

        self.seen_funcs = set()
        self.naming_map = dict()  # key:string, val:dict(key: func, val: index)
        self.input_name_map = dict()    # key:variable, val:string
        self.constant_input_name_map = dict()    # key:variable, val:string

    def _get_layer_name(self, layer):
        """Generate layer name like "Convolution2DFunction-10-2".

        The first number means rank of the layer (depth from the top),
        and the second number is for preventing duplication
        (different layer objects can have the same rank)
        :param layer: Function object
        :return: string unique to functions
        """
        # e.g. F.array.broadcast.BroadcastTo -> BroadcastTo
        name = type(layer).__name__

        # e.g. BroadcastTo-0
        label = '{}-{}'.format(name, layer.rank)
        if hasattr(layer, '_MarkPrefixHook__chainer_trt_prefix'):
            pfx = layer._MarkPrefixHook__chainer_trt_prefix
            label = '{}-{}'.format(pfx, label)

        if label not in self.naming_map:
            self.naming_map[label] = dict()

        if layer not in self.naming_map[label].keys():
            self.naming_map[label][layer] = len(self.naming_map[label]) + 1
        return '{}-{}'.format(label, self.naming_map[label][layer])

    def get_source_name(self, input_):
        """Get name of the layer

        :param input_: chainer.{Variable,VariableNode} inside the
           computational graph
        :return: Found or determined name of its creator
        """
        parent = input_.creator
        if parent is None:      # This is an input layer
            return self._dump_input_and_get_name(input_)
        return self._get_layer_name(parent)

    def _dump_input_and_get_name(self, input_):
        if input_ in self.input_name_map:
            return self.input_name_map[input_]
        if input_ in self.constant_input_name_map:
            return self.constant_input_name_map[input_]

        # (first condition):
        # For compatibility, every inputs are treated as user inputs,
        # if nothing is registered
        if len(self.registered_user_input_names) != 0 and \
                input_ not in self.registered_user_input_names:
            input_type, rank = 'ConstantInput', self.RANK_CONSTANT_INPUT
            n_const_inputs = len(self.constant_input_name_map)
            input_name = '{}-{}'.format(input_type, n_const_inputs)
            self.constant_input_name_map[input_] = input_name
        else:
            input_type, rank = 'input', self.RANK_INPUT
            if input_ not in self.registered_user_input_names or \
                    self.registered_user_input_names[input_] is None:
                n_inputs = len(self.input_name_map)
                input_name = '{}-{}'.format(input_type, n_inputs)
            else:
                input_name = self.registered_user_input_names[input_]
            self.input_name_map[input_] = input_name

        input_params = {
            'type': input_type,
            'name': input_name,
            'rank': rank,       # Make sure ConstantInputs come after Inputs
            'shape': list(input_.shape[1:]),    # Eliminate batch dim
        }

        if self.verbose or input_type == 'ConstantInput':
            filename = '{}.tensor'.format(input_name)
            input_params['input_tensor'] = filename
            fn = '{}/{}'.format(self.dst_path, filename)
            ModelRetriever._save_tensor(fn, input_.data)
        self.layers.append(input_params)

        return input_name

    def _merge_dicts(self, *dicts):
        # to support py2 dict merge ({**d1, **d2} is not supported by py2)
        return {k: v for d in dicts for (k, v) in d.items()}

    def _dump_conv_deconv(self, func, initial_param):
        input_, W = func.inputs[:2]

        # check if it has bias
        if len(func.inputs) == 3:
            b = func.inputs[2]
        else:
            b = chainer.Variable(np.array([], dtype=np.float32))

        parent_layer_name = self.get_source_name(input_)
        dx = None
        dy = None

        groups = func.groups if hasattr(func, 'groups') else 1

        if type(func) is Convolution2DFunction:
            n_out, n_in, kh, kw = W.shape
        elif type(func) is Deconvolution2DFunction:
            n_in, n_out, kh, kw = W.shape
            n_out = n_out * groups

        if hasattr(func, "dx") and hasattr(func, "dy"):
            dx = func.dx
            dy = func.dy

        param = {
            'source': parent_layer_name, 'n_out': n_out,
            'kernel_w': kw, 'kernel_h': kh,
            'pad_w': func.pw, 'pad_h': func.ph,
            'stride_x': func.sx, 'stride_y': func.sy,
            'groups': groups
        }

        if dx is not None and dy is not None:
            param["dilation_x"] = dx
            param["dilation_y"] = dy

        if hasattr(func, 'cover_all'):  # Conv layers can have cover_all param
            param['cover_all'] = func.cover_all
        bias = b.data
        if bias is None or bias.size == 0:
            bias = np.zeros(n_out)
        attr = {'kernel': W.data, 'bias': bias}
        return self._merge_dicts(initial_param, param), attr

    def _dump_bn(self, func, initial_param):
        # http://docs.chainer.org/en/latest/_modules/chainer/functions/normalization/batch_normalization.html
        input_, gamma, beta, mean, var = func.inputs
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name, 'eps': func.eps}
        if beta.data is None or beta.data.size == 0:
            beta.data = np.zeros(mean.data.size)
        attr = {
            'mean': mean.data, 'var': var.data,
            'gamma': gamma.data, 'beta': beta.data
        }
        return self._merge_dicts(initial_param, param), attr

    def _dump_shift(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {
            'source': parent_layer_name, 'kw': func.kw, 'kh': func.kh,
            'dx': func.dx, 'dy': func.dy
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_lrn(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {
            'source': parent_layer_name,
            'n': func.n, 'k': func.k, 'alpha': func.alpha, 'beta': func.beta
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_activation(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name}
        return self._merge_dicts(initial_param, param), None

    def _dump_leaky_relu(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name, 'slope': func.slope}
        return self._merge_dicts(initial_param, param), None

    def _dump_concat(self, func, initial_param):
        parent_layer_names = [self.get_source_name(input_)
                              for input_ in func.inputs]
        param = {
            'sources': parent_layer_names,
            'axis': func.axis - 1  # eliminate batch dim
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_copy(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name}
        return self._merge_dicts(initial_param, param), None

    def _dump_softmax(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name}
        return self._merge_dicts(initial_param, param), None

    def _dump_reshape(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)

        input_shape = input_.shape[1:]  # eliminate batch dim
        shape = list(func.shape[1:])    # eliminate batch dim
        if -1 in shape:
            assert (shape.count(-1) == 1)
            n_element = -np.prod(shape)
            real_n_element = np.prod(input_shape)
            shape[shape.index(-1)] = int(real_n_element // n_element)

        in_sum = np.prod(input_shape)
        out_sum = np.prod(shape)
        assert in_sum == out_sum, "shape doesn't match"

        param = {'source': parent_layer_name, 'shape': shape}
        return self._merge_dicts(initial_param, param), None

    def _dump_pooling(self, func, initial_param):
        input_ = func.inputs[0]
        parent_layer_name = self.get_source_name(input_)
        param = {
            'source': parent_layer_name,
            'window_width': func.kw, 'window_height': func.kh,
            'stride_x': func.sx, 'stride_y': func.sy,
            'pad_w': func.pw, 'pad_h': func.ph,
            'cover_all': func.cover_all
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_linear(self, func, initial_param):
        input_, W = func.inputs[:2]
        if len(func.inputs) == 3:
            b = func.inputs[2]
        else:
            b = np.zeros(shape=W.shape[0], dtype=W.data.dtype)
            b = chainer.Variable(b)

        parent_layer_name = self.get_source_name(input_)

        # Workaround: in Chainer input tensor to Linear layer is reshaped
        # so that ndim becomes 2 before applying mat dot.
        # From Chainer V3 this is done for the input Variable
        # instead of xp.array.
        # So in computational graph, a new Reshape node is inserted.
        # https://github.com/chainer/chainer/blob/v3.0.0/chainer/functions/connection/linear.py#L111
        # This makes computational graph incompatible with TensorRT
        # (TensorRT always require NCHW (ndim=4)).
        # So, in this workaround, if creator of Linear is reshape node,
        # it skips that node.
        # This workaround also resolves the problem stated in
        # https://github.com/chainer/chainer/pull/3474
        parent_layer_type = type(input_.creator).__name__
        if parent_layer_type == 'Reshape':
            input_, = input_.creator.inputs
            old_parent_layer_name = parent_layer_name
            parent_layer_name = self.get_source_name(input_)
            if self.verbose:
                print('Skipping {}, source of {} is now {}'
                      .format(old_parent_layer_name, initial_param["name"],
                              parent_layer_name))

        param = {'source': parent_layer_name, 'n_out': W.shape[0]}
        attr = {'kernel': W.data, 'bias': b.data}
        return self._merge_dicts(initial_param, param), attr

    def _dump_eltw(self, func, initial_param):
        parent_layer_names = [self.get_source_name(input_)
                              for input_ in func.inputs]
        param = {'sources': parent_layer_names}
        return self._merge_dicts(initial_param, param), None

    def _dump_const_eltw(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        param = {'source': parent_layer_name}
        attr = {'constant': func.value}
        return self._merge_dicts(initial_param, param), attr

    def _dump_broadcast(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        # Eliminate batch dim
        # [Note] The key name "output_shape" is already used in verbose mode,
        # so we should use other key name.
        param = {'source': parent_layer_name,
                 'in_shape': list(input_.shape[1:]),
                 'out_shape': list(func._shape[1:])}
        return self._merge_dicts(initial_param, param), None

    def _dump_dropout(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        param = {
            'source': parent_layer_name, 'dropout_ratio': func.dropout_ratio
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_transpose(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        axes = [t - 1 for t in func.axes[1:]]   # Eliminate batch dim
        param = {'source': parent_layer_name, 'axes': axes}
        return self._merge_dicts(initial_param, param), None

    def _dump_resize(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        param = {
            'source': parent_layer_name,
            'n_channels': input_.shape[1],
            'input_hw': [input_.shape[2], input_.shape[3]],
            'output_hw': [func.out_H, func.out_W]
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_getitem(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)

        # --- type check ---
        for s in func.slices[1:]:
            if s is None:
                raise TypeError(
                    'None is used in getitem, but it is not supported now.'
                    'You may explicitly use `reshape` method.')

        slices = [[s.start, s.stop, s.step] if isinstance(s, slice) else s
                  for s in func.slices[1:]]   # without batch dim
        slices += [[None, None, None]] * (len(input_.shape) - len(func.slices))
        param = {'source': parent_layer_name, 'slices': slices}
        return self._merge_dicts(initial_param, param), None

    def _dump_where(self, func, initial_param):
        parent_layer_names = [self.get_source_name(input_)
                              for input_ in func.inputs]
        param = {'sources': parent_layer_names}
        return self._merge_dicts(initial_param, param), None

    def _dump_unary(self, func, initial_param):
        initial_param['type'] = 'Unary'
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        if type(func) == F.math.exponential.Exp:
            op = 'exp'
        param = {'source': parent_layer_name, 'operation': op}
        return self._merge_dicts(initial_param, param), None

    def _dump_argmax(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        if func.axis != 1:
            msg = "Only axis=1 is supported for Argmax ({} is specified)"\
                .format(func.axis)
            raise RuntimeError(msg)
        param = {'source': parent_layer_name, 'shape': input_.shape[1:]}
        return self._merge_dicts(initial_param, param), None

    def _dump_resize_argmax(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)

        if func.argmax.axis != 1:
            msg = "Only axis=1 is supported for Argmax ({} is specified)"\
                .format(func.argmax.axis)
            raise RuntimeError(msg)

        param = {
            'source': parent_layer_name,
            'n_channels': input_.shape[1],
            'input_hw': [input_.shape[2], input_.shape[3]],
            'output_hw': [func.resize.out_H, func.resize.out_W]
        }
        return self._merge_dicts(initial_param, param), None

    def _dump_sum(self, func, initial_param):
        input_, = func.inputs
        parent_layer_name = self.get_source_name(input_)
        if func.axis != (1,):
            msg = "Only axis=(1,) is supported for Sum ({} is specified)"\
                .format(func.axis)
            raise RuntimeError(msg)
        param = {'source': parent_layer_name, 'shape': input_.shape[1:]}
        return self._merge_dicts(initial_param, param), None

    def _dump_linear_interpolate(self, func, initial_param):
        parent_layer_names = [self.get_source_name(input_)
                              for input_ in func.inputs]
        param = {'sources': parent_layer_names}
        return self._merge_dicts(initial_param, param), None

    def _dump_matmul(self, func, initial_param):
        parent_layer_names = [self.get_source_name(input_)
                              for input_ in func.inputs]
        param = {
            'sources': parent_layer_names,
            'transa': func.transa,
            'transb': func.transb
        }

        if func.transc:
            msg = "Currently only transc=False is supported for" \
                  "MatMul ({} is specified)".format(func.transc)
            raise RuntimeError(msg)
        if func.dtype is not None:
            msg = "Currently only dtype=None is supported for " \
                  "MatMul ({} is specified)".format(func.dtype)
            raise RuntimeError(msg)
        return self._merge_dicts(initial_param, param), None

    def _raise_non_test_mode_func(self, _, initial_param):
        msg = "{} is found, which should appear only in training mode. " \
              "Have you ran forward pass with test mode by doing " \
              "\"with chainer.using_config('train', False):\"?"\
            .format(initial_param['type'])
        raise RuntimeError(msg)

    def _raise_unsupported(self, func, initial_param):
        raise "Layer type {} is not supported".format(initial_param['type'])

    _dump_func_map = {
        Convolution2DFunction: _dump_conv_deconv,
        Deconvolution2DFunction: _dump_conv_deconv,
        F.normalization.batch_normalization.FixedBatchNormalization: _dump_bn,
        F.normalization.local_response_normalization.LocalResponseNormalization: _dump_lrn,     # NOQA
        F.activation.relu.ReLU: _dump_activation,
        F.activation.sigmoid.Sigmoid: _dump_activation,
        F.activation.tanh.Tanh: _dump_activation,
        F.activation.leaky_relu.LeakyReLU: _dump_leaky_relu,
        F.array.concat.Concat: _dump_concat,
        F.array.copy.Copy: _dump_copy,
        F.activation.softmax.Softmax: _dump_softmax,
        F.array.reshape.Reshape: _dump_reshape,
        F.pooling.max_pooling_2d.MaxPooling2D: _dump_pooling,
        F.pooling.average_pooling_2d.AveragePooling2D: _dump_pooling,
        F.connection.linear.LinearFunction: _dump_linear,
        Add: _dump_eltw, Sub: _dump_eltw, Mul: _dump_eltw, Div: _dump_eltw,
        MulConstant: _dump_const_eltw, AddConstant: _dump_const_eltw,
        SubFromConstant: _dump_const_eltw, DivFromConstant: _dump_const_eltw,
        F.math.maximum.Maximum: _dump_eltw,
        F.math.minimum.Minimum: _dump_eltw,
        F.array.broadcast.BroadcastTo: _dump_broadcast,
        F.noise.dropout.Dropout: _dump_dropout,
        F.array.transpose.Transpose: _dump_transpose,
        F.array.resize_images.ResizeImages: _dump_resize,
        F.array.get_item.GetItem: _dump_getitem,
        F.array.where.Where: _dump_where,
        F.math.exponential.Exp: _dump_unary,
        F.math.minmax.ArgMax: _dump_argmax,
        F.math.matmul.MatMul: _dump_matmul,
        chainer_trt.functions.ResizeArgmax: _dump_resize_argmax,
        F.math.sum.Sum: _dump_sum,
        F.math.linear_interpolate.LinearInterpolate: _dump_linear_interpolate,
        F.connection.shift.Shift: _dump_shift,

        # These layers should not appear in test mode
        F.normalization.batch_normalization.BatchNormalization: _raise_non_test_mode_func     # NOQA
    }

    _name_conversion_table = {
        'FixedBatchNormalization': 'BatchNormalizationFunction'
    }

    def add_dump_function(self, type, dump_func):
        """Add custom dump function

        This is an interface to extend ModelRetriever to make it possible to
        dump arbitrary type of layer.

        :param type:
          subclass of chainer.Function
        :param dump_func:
          method to dump the chainer.Function object.
          Its signature should be
          `dump_func(model_retriever, function_object, initial_params)`
          Its return value should be a tuple of:
          * parameter dictionary saved to model.json
          * wight dictionary to save, or None if nothing to save
        """
        assert issubclass(type, chainer.Function) or \
            issubclass(type, chainer.FunctionNode)
        assert hasattr(dump_func, '__call__')
        self._dump_func_map[type] = dump_func

    @classmethod
    def _save_tensor(cls, filename, ar):
        if ar is None:
            return
        with open(filename, 'wb') as f:
            if type(ar) in [float, int]:
                ar = np.array([ar])
            if ar.dtype.itemsize > 4:
                ar = ar.astype(np.float32)
            f.write('{}\n'.format(ar.dtype.itemsize).encode())

            s = ar.shape
            s = ",".join([str(t) for t in s])
            f.write('{}\n'.format(s).encode())

            device = chainer.cuda.get_device(ar)
            if type(device) is not chainer.cuda.DummyDeviceType:
                # It's on GPU
                ar = chainer.cuda.to_cpu(ar)
            ar = np.ascontiguousarray(ar)
            if hasattr(ar.data, 'tobytes'):
                f.write(ar.data.tobytes())  # py3
            else:
                f.write(ar.tobytes())   # py2

    def _dump_function_object(self, func, func_output):
        assert isinstance(func, chainer.function.Function) or \
            isinstance(func, chainer.function_node.FunctionNode)
        layer_name = self._get_layer_name(func)
        layer_type = type(func).__name__
        if layer_type in self._name_conversion_table:
            layer_type = self._name_conversion_table[layer_type]
        if self.verbose:
            print(layer_name)

        # Find correct type of layer dumper and call it
        initial_param = {
            'type': layer_type, 'name': layer_name, 'rank': func.rank
        }
        dump = self._dump_func_map[type(func)]
        layer_param, weights = dump(self, func, initial_param)

        # Check if DLA is specified
        if hasattr(func, '_chainer_trt_enable_dla'):
            layer_param['dla'] = True

        # Save weights to binary file if exist
        if weights is not None:
            for name, values in weights.items():
                if values is not None:
                    weight_fn = '{}_{}.weights'.format(layer_name, name)
                    layer_param['{}_weights_file'.format(name)] = weight_fn
                    weight_path = '{}/{}'.format(self.dst_path, weight_fn)
                    ModelRetriever._save_tensor(weight_path, values)

        if self.verbose:
            out_fn = '{}_output.tensor'.format(layer_name)
            out_path = '{}/{}'.format(self.dst_path, out_fn)
            ModelRetriever._save_tensor(out_path, func_output.data)
            layer_param['input_shapes'] = [input_.shape
                                           for input_ in func.inputs]
            layer_param['input_types'] = [input_.dtype.name
                                          for input_ in func.inputs]
            layer_param['output_shape'] = func_output.shape
            layer_param['output_type'] = func_output.dtype.name
            layer_param['output_tensor'] = out_fn

        if hasattr(func, '_TracebackHook__chainer_trt_traceback'):
            layer_param['traceback'] = \
                func._TracebackHook__chainer_trt_traceback

        self.layers.append(layer_param)

    def __call__(self, var, name=None):
        """Dump computational graph from an output variable.

        If you have mutliple outputs from a NN, call __call__ for each of them.
        The duplicated part of the computational graph is properly handled
        by model_retriever.
        :param var: chainer.Variable whose creator must not be None
        :param name: Specify name of the output. Default is None (auto naming)
        :return: self
        """
        if type(var) is list:
            msg = "Specifying a list of variables to __call__ is supported " \
                  "but not recommended. Please consider calling one by one"
            warnings.warn(msg)
            if name is not None:
                msg = "name argument is specified but ignored, " \
                      "as list of variables is given"
                warnings.warn(msg)
            for v in var:
                self(v)
            return self

        if var.creator is None:
            raise ValueError("The specified variable has no creator.")

        self.output_variables.append(var)

        output_name = self._get_layer_name(var.creator)
        self.output_names[output_name] = output_name if name is None else name

        if var.creator in self.seen_funcs:
            return self

        funcs = [(var.creator, var)]
        self.seen_funcs.add(var.creator)

        # retrieval loop
        while funcs:
            func, func_output = funcs.pop(0)
            self._dump_function_object(func, func_output)

            inputs = func.inputs
            for _input in inputs:
                creator = _input.creator
                if creator is not None and creator not in self.seen_funcs:
                    assert isinstance(creator, chainer.function.Function) or \
                        isinstance(creator, chainer.function_node.FunctionNode)
                    funcs.append((creator, _input))
                    self.seen_funcs.add(creator)
        return self

    def preprocess_caffemodel(self, net):
        """Apply preprocess to a CaffeFunction object.

        It merges a Scale layer that runs right after BN layer.
        Caffe BN layer only has normalization parameters,
        which are mean and var, but BN operation actually needs
        scale and shift parameter.
        That process is done by succeeding Scale layer.
        In chainer both normalization and scale/shift parameters
        can be stored in BN object. This function eliminates Scale layer
        and bring its parameter into BN layer.
        This function has to be called before running net.__call__().

        Args:
            net (chainer.links.caffe.caffeFunction): CaffeFunction object to
                be retrieved. Layers will be modified.
        """
        BatchNormalization = \
            chainer.links.normalization.batch_normalization.BatchNormalization
        Scale = chainer.links.connection.scale.Scale

        if self.verbose:
            print("Applying caffemodel preprocessing to the network")

        # Algorithm:
        # There are 2 patterns of mergeable BN+Scale.
        # 1. Both BN and Scale refer the same src (ResNet caffemodel pattern)
        # 2. Scale refers BN (DenseNet caffemodel patten)

        # for pattern 1, finding common source (str => index)
        bn_scale_common_sources = dict()
        bns = dict()
        scale_indices = []
        for i, (func_name, bottoms, _) in enumerate(net.layers):
            if not hasattr(net, func_name):
                # function that has no parameter like ReLU doesn't exist
                # in `net` as an attr but BN and Scale do
                continue
            layer = getattr(net, func_name)

            # Find sources that are referred by both BN and Scale layers
            for src_name in bottoms:
                # In net.layers, BN always comes first and Scale comes next
                if type(layer) is BatchNormalization:
                    # For patten 1: Mark the source `bottom` is referred by BN
                    bn_scale_common_sources[src_name] = i

                    # For patten 2: Mark the layer is BN
                    bns[func_name] = i
                elif type(layer) is Scale:
                    if bn_scale_common_sources.get(src_name) is not None:
                        # Pattern 1 found:
                        # This source `bottom` is already marked;
                        # That is obviously BN
                        i = bn_scale_common_sources[src_name]
                        bn_layer_name, _, _ = net.layers[i]
                        bn = getattr(net, bn_layer_name)
                        bn.gamma = chainer.Parameter(layer.W.data)
                        if hasattr(layer, 'bias'):
                            bn.beta = chainer.Parameter(layer.bias.b.data)
                        scale_indices.append(i)
                        if self.verbose:
                            print('Mergeable BN detected '
                                  '(BN:{}/Scale:{} both refer {})'
                                  .format(bn_layer_name, func_name, src_name))
                    elif src_name in bns:
                        # Pattern 2 found: This scale layers refers BN layer!!
                        bn = getattr(net, src_name)
                        bn.gamma = chainer.Parameter(layer.W.data)
                        if hasattr(layer, 'bias'):
                            bn.beta = chainer.Parameter(layer.bias.b.data)
                        scale_indices.append(i)
                        if self.verbose:
                            print('Mergeable BN detected '
                                  '(Scale:{} refers BN:{})'
                                  .format(func_name, src_name))

        # Remove scale layers that are no longer necessary
        for i in sorted(scale_indices, reverse=True):
            del net.layers[i]

    def _gc(self):
        """Remove layers that are not referred from any other layers

        This is needed when the workaround in _dump_linear is applied
        (dangling Reshape will occur)
        """
        removed_layer_names = set()
        while True:
            # mark
            used_sources = set(self.output_names.keys())
            for layer in self.layers:
                if 'source' in layer:
                    used_sources.add(layer['source'])
                elif 'sources' in layer:
                    for name in layer['sources']:
                        used_sources.add(name)

            if len(used_sources) == len(self.layers):
                break

            # sweep
            new_layers = [layer for layer in self.layers
                          if layer['name'] in used_sources]
            current_layer_names = set([t['name']for t in self.layers])
            new_layer_names = set([t['name'] for t in new_layers])
            removed_layer_names |= current_layer_names - new_layer_names
            self.layers = new_layers

        if self.verbose:
            print('The following layers are removed by GC: {}'
                  .format(", ".join(removed_layer_names)))

    def _layer_sort_key(self, layer):
        input_idx = 0
        if layer['type'] == 'input':
            # Input layers have to be the same order as input name list
            # (which is the order of user specification by register_inputs)
            input_idx = self._get_input_names().index(layer['name'])

        return layer['rank'], input_idx

    def _get_input_names(self):
        if len(self.registered_user_input_names):
            # When user inputs are registered,
            # bring layer names in the order of registration.
            # (input_name_map is ordered by when each output is found
            # in the computational graph)
            # And input that is registered but not used shouldn't appear.
            input_names = []
            for x, name in self.registered_user_input_names.items():
                if x in self.input_name_map:
                    input_names.append(name)
        else:
            input_names = list(self.input_name_map.values())

        return input_names

    def generate_json_source(self):
        self._gc()

        # Write prototxt in reverse order
        # (layers close to input should be in top, similar to topological sort)
        self.layers.sort(key=self._layer_sort_key)

        return {
            'inputs': self._get_input_names(),
            'outputs': [[k, v] for k, v in self.output_names.items()],
            'layers': self.layers
        }

    def save(self):
        model_json = self.generate_json_source()
        json_path = '{}/model.json'.format(self.dst_path)
        json.dump(model_json, open(json_path, 'wt'),
                  indent=(2 if self.verbose else None), cls=JSONEncoderEX)

        # Save computational graph (for ease of debug)
        if self.verbose:
            g = build_computational_graph(self.output_variables)
            graph_path = '{}/computational_graph.dot'.format(self.dst_path)
            with open(graph_path, 'w') as o:
                o.write(g.dump())

    def register_inputs(self, x, name=None):
        if type(x) in [list, tuple]:
            if name is not None:
                msg = "name={} is ignored because x is a collection." \
                    .format(name)
                warnings.warn(msg)
            for _x in x:
                self.register_inputs(_x)
        elif type(x) is chainer.Variable:
            self.register_inputs(x.node, name=name)
        elif type(x) is chainer.variable.VariableNode:
            self.registered_user_input_names[x] = name
        else:
            msg = "Please specify a Variable, a VariableNode or list of them"
            raise TypeError(msg)
        return self
