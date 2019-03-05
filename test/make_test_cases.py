import argparse
import collections
import fnmatch
import json

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

import chainer_trt

generators = list()

# Use this type to describe acceptable error for each test case
error = collections.namedtuple('error', ['fp32', 'fp16', 'int8'])
default_err_rough = error(0.001, 0.01, 0.02)
default_err_strict = error(1e-6, 0.001, 0.01)
default_err_zero = error(0, 0, 0)


def generator(errs=default_err_strict, batch_sizes=(1, 1024),
              name_suffix='', int8_calib_cache='',
              custom_dump_functions=dict()):
    """Use this decorator for test case generation function

    By using @generator() for your generation function, this script
    recognizes it and generates various patterns of test cases.
    Be noted that it has to be `@generator()` even if you don't specify
    any argument, rather than `@generator` (no braces)

    :param errs:
      Acceptable aboslute error, for each fp32/fp16/int8 mode.
      By default it is default_err_strict. There are predefined errors,
      which are default_err_rough and default_err_zero, but you can specify
      arbitrary error by using `error` named tuple.
    :param batch_sizes:
      Batch sizes to test. Default (1 and 1024) is enough.
    :param name_suffix:
      Additional suffix to prevent name conflict.
      Suffix should be unique in the test cases for the same type.
      Use it with @generator_set decorator (otherwise you don't need)
    """
    def decorator(func):
        assert len(errs) == 3
        generators.append((func, errs, batch_sizes, name_suffix,
                          int8_calib_cache, custom_dump_functions))
    return decorator


def generator_set(func):
    """Use this decorator when you wil generate multiple test cases

    You can define a function that defines lots of generator function.
    Such a meta-generator has to use this @generator_set decorator.

    Each generator should also use @generator.
    """
    func()


def rand(s, var=True, bias=0):
    x = np.random.rand(*s).astype(np.float32) + bias
    return chainer.Variable(x) if var else x


def randint(n, s, var=True):
    x = np.random.randint(n, size=s)
    return chainer.Variable(x) if var else x


def ones(s, var=True):
    x = np.ones(s).astype(np.float32)
    return chainer.Variable(x) if var else x


###############################################################################
# Define test case generation functions below
###############################################################################


@generator(errs=default_err_rough)
def linear():
    x = rand((1, 100, 1, 1))
    W = rand((200, 100), var=False)
    b = rand((200,), var=False)
    y = F.linear(x, W, b=b)
    y = y.reshape(1, 200, 1, 1)
    return {'input': x}, {'out': y}


@generator_set
def elementwises():
    M = F.math.basic_math
    ops = [M.add, M.mul, M.sub, M.div]

    for op in ops:
        @generator(name_suffix='_' + str(op.__name__))
        def elementwise():
            x1 = rand((1, 3, 8, 8))
            x2 = rand((1, 3, 8, 8))
            y = op(x1, x2)
            return {'input-0': x1, 'input-1': x2}, {'out': y}

        @generator(name_suffix='_' + str(op.__name__))
        def constant_elementwise():
            x1 = rand((1, 3, 8, 8))
            x2 = rand((1, 3, 8, 8))
            y = op(x1, x2)
            return {'input-0': x1}, {'out': y}

        @generator(name_suffix='_' + str(op.__name__))
        def constant_elementwise_scalar():
            x1 = rand((1, 3, 8, 8))
            x2 = 5
            y = op(x1, x2)
            return {'input-0': x1}, {'out': y}


@generator_set
def activations():
    A = F.activation
    ops = [A.relu.relu, A.sigmoid.sigmoid, A.tanh.tanh]

    for op in ops:
        @generator(name_suffix='_' + str(op.__name__))
        def activation():
            x = rand((1, 3, 8, 8))
            y = op(x)
            return {'input': x}, {'out': y}

    for slope in [0.2, 0.4]:
        @generator(name_suffix='_' + str(slope).replace('.', '_'))
        def leaky_relu():
            x = rand((1, 3, 8, 8))
            y = F.leaky_relu(x, slope=0.2)
            return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def exp():
    x = rand((1, 8, 8, 8))
    y = F.exp(x)
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def log():
    x = rand((1, 8, 8, 8), bias=1e-10)
    y = F.log(x)
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def sqrt():
    x = rand((1, 8, 8, 8))
    y = F.sqrt(x)
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def abs():
    x = rand((1, 8, 8, 8), bias=-0.5)
    y = F.absolute(x)
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def neg():
    x = rand((1, 8, 8, 8), bias=-0.5)
    y = -x
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def recip():
    x = rand((1, 8, 8, 8), bias=1e-10)
    y = 1 / x
    return {'input': x}, {'out': y}


@generator(errs=default_err_strict)
def reshape():
    x = rand((1, 8, 8, 8))
    y = F.reshape(x, (1, 1, 64, 8))
    return {'input': x}, {'out': y}


@generator_set
def slices():
    @generator(errs=default_err_zero)
    def slice():
        x = rand((1, 3, 10, 10))
        y = x[:, 1:, 3:5, 3:6:2]
        return {'input': x}, {'out': y}

    @generator(errs=default_err_zero)
    def slice_neg_step():
        x = rand((1, 3, 10, 10))
        y = x[:, :2, 5:3:-1, -5:-3:1]
        return {'input': x}, {'out': y}

    @generator(errs=default_err_zero)
    def slice_scalar():
        x = rand((1, 3, 10, 10))
        y = x[:, 1:, 3:5, 3:6:2]   # replace slice
        return {'input': x}, {'out': y}


@generator(errs=default_err_strict)
def transpose():
    x = rand((1, 3, 10, 10))
    y = x.transpose(0, 2, 3, 1)
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def constant_input():
    x0 = rand((1, 3, 10, 10))
    x1 = rand((1, 3, 10, 10))
    x2 = rand((1, 3, 10, 10))
    x3 = rand((1, 3, 10, 10))
    c0 = rand((1, 3, 10, 10), var=False)
    c1 = rand((1, 3, 10, 10), var=False)
    c2 = rand((1, 3, 10, 10), var=False)
    c3 = rand((1, 3, 10, 10), var=False)

    # Run operation with random order
    y = x1 * x3 / x2 - x0

    # constant inputs will appear in ascending order while retrieval
    y = y + c3
    y = y / c2
    y = y * c1
    y = y - c0

    return {'input-0': x0, 'input-1': x1, 'input-2': x2, 'input-3': x3}, \
           {'out': y}


@generator_set
def minimum_maximum():
    for op in [F.minimum, F.maximum]:
        @generator(name_suffix='_' + str(op.__name__))
        def minimum_maximum():
            x0 = rand((1, 3, 8, 8))
            x1 = rand((1, 3, 8, 8))
            y = op(x0, x1)
            return {'input-0': x0, 'input-1': x1}, {'out': y}


@generator()
def shift():
    x = rand((1, 9, 10, 10))
    y = F.shift(x, ksize=3, dilate=1)
    return {'input': x}, {'out': y}


@generator()
def broadcast_to():
    x = rand((1, 3, 1, 1))
    y = F.broadcast_to(x, (1, 3, 8, 8))
    return {'input': x}, {'out': y}


@generator_set
def pooling():
    P = F.pooling
    ops = [P.max_pooling_2d.max_pooling_2d,
           P.average_pooling_2d.average_pooling_2d]

    for op in ops:
        @generator(name_suffix='_' + str(op.__name__))
        def pooling():
            x = rand((1, 3, 21, 21))
            y = op(x, ksize=2, stride=2)
            return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def resize():
    x = rand((1, 3, 32, 32))
    y = F.resize_images(x, (45, 45))
    return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def resize_argmax():
    x = rand((1, 3, 32, 32))
    y = chainer_trt.functions.resize_argmax(x, (128, 128), axis=1)
    # needed reshape, due to the same reason as argmax
    y = y.reshape(1, 1, 128, 128)
    return {'input': x}, {'out': y}


@generator_set
def concat():
    x0 = rand((1, 3, 10, 10))
    x1 = rand((1, 3, 10, 10))

    @generator(errs=default_err_zero)
    def concat_default():
        y = F.concat([x0, x1])
        return {'input-0': x0, 'input-1': x1}, {'out': y}

    @generator(errs=default_err_zero)
    def concat_axis():
        y = F.concat([x0, x1], axis=2)
        return {'input-0': x0, 'input-1': x1}, {'out': y}


@generator_set
def conv_deconv():
    x = rand((1, 16, 10, 10))

    @generator(errs=default_err_rough)
    def convolution():
        conv = L.Convolution2D(16, 32, ksize=3, pad=1)
        y = conv(x)
        return {'input': x}, {'out': y}

    @generator(errs=default_err_rough)
    def grouped_conv():
        conv = L.Convolution2D(16, 32, ksize=3, pad=1, groups=4)
        y = conv(x)
        return {'input': x}, {'out': y}

    @generator(errs=default_err_rough)
    def dilated_conv():
        conv = L.Convolution2D(16, 32, ksize=3, pad=2, dilate=2)
        y = conv(x)
        return {'input': x}, {'out': y}

    @generator(errs=default_err_rough)
    def deconvolution():
        conv = L.Deconvolution2D(16, 32, ksize=3, pad=1)
        y = conv(x)
        return {'input': x}, {'out': y}

    @generator(errs=default_err_rough)
    def grouped_deconv():
        conv = L.Deconvolution2D(16, 32, ksize=3, pad=1, groups=4)
        y = conv(x)
        return {'input': x}, {'out': y}


@generator(errs=default_err_rough)
def batchnorm():
    x = rand((1, 3, 32, 32))
    gamma = rand((3,))
    beta = rand((3,))
    mean = rand((3,))
    var = rand((3,))
    y = F.fixed_batch_normalization(x, gamma, beta, mean, var)
    return {'input': x}, {'out': y}


@generator()
def argmax():
    x = rand((1, 16, 8, 8))
    y = F.argmax(x, axis=1)
    # argmax returns (1, 8, 8) but chainer_trt's argmax is (1, 1, 8, 8)
    y = y.reshape(1, 1, 8, 8)
    return {'input': x}, {'out': y}


@generator()
def linear_interpolate():
    x0 = rand((1, 2, 3, 4))
    x1 = rand((1, 2, 3, 4))
    x2 = rand((1, 2, 3, 4))
    y = F.linear_interpolate(x0, x1, x2)
    return {'input-0': x0, 'input-1': x1, 'input-2': x2}, {'out': y}


@generator_set
def sum():
    @generator(errs=default_err_rough)
    def sum():
        x = rand((1, 16, 10, 10))
        y = F.sum(x, axis=1)
        return {'input': x}, {'out': y}

    @generator(errs=default_err_rough)
    def sum_another_case():
        x = rand((1, 16, 2, 2, 2))
        y = F.sum(x, axis=1)
        return {'input': x}, {'out': y}


# With current chainer, where export doesn't work due to
# https://github.com/chainer/chainer/commit/3cc8d43fb82cceebaef
# @generator()
def where():
    c = chainer.Variable(randint(2, (1, 2, 3, 4), var=False) == 0)
    x0 = rand((1, 2, 3, 4))
    x1 = rand((1, 2, 3, 4))
    y = F.where(c, x0, x1)
    return {'input-c': c, 'input-0': x0, 'input-1': x1}, {'out': y}


@generator()
def copy():
    x = rand((1, 2, 3, 4))
    y = F.copy(x, -1) - 0
    # A dummy layer `-0` is inserted because our implementation of copy plugin
    # reuses input tensor as the layer output.
    # (a tensor cannot be both input and output of a network)
    # when using copy inside a network with multiple layers,
    # no need to insert any dummy operations.
    return {'input': x}, {'out': y}


@generator_set
def matmul():
    x0 = rand((1, 2, 3, 4))
    x1 = rand((1, 2, 4, 5))
    for transa in [True, False]:
        for transb in [True, False]:
            @generator(errs=default_err_rough,
                       name_suffix='_transa{}_transb{}'.format(transa, transb))
            def matmul():
                y = F.matmul(x0, x1, transa=transa, transb=transb)
                return {'input-0': x0, 'input-1': x1}, {'out': y}


# Multiple output test

class Increment3(chainer.FunctionNode):
    def forward_cpu(self, inputs):
        x = inputs[0]
        return (x + 1, x + 2, x + 3)


def dump_increment3(model_retriever, func, initial_params):
    source = model_retriever.get_source_name(func.inputs[0])
    initial_params['source'] = source
    return initial_params, None     # No weights


@generator(custom_dump_functions={Increment3: dump_increment3})
def multiple_outputs():
    x = rand((1, 3, 10, 10))
    f = Increment3()
    y0, y1, y2 = f.apply((x,))
    return {'input': x}, {'y0': y0, 'y1': y1, 'y2': y2}


@generator_set
def regression_imagenets():
    """Generates ImageNet models for regression tests

    The exports made by this fixture is (and should be) compatible with
    example_imagenet/imagenet_tensorrt_builder, which means what is done in
    this part is same as example_imagenet/dump_chainer.py.
    This is because calibration cache file (as a part of test fixture) is
    generated by the builder like the following way.

    ```
    % imagenet_tensorrt_builder \
        -i test/fixtures/model/regression_imagenet_VGG16Layers \
        -o /dev/null --mode int8 --calib ${calib_data} \
        --out-cache test/fixtures/regression/VGG16Layers_int8_calib_cache.dat
    ```

    Regeneration of ImageNet calibration cache fixture is needed
    in case you modify ModelRetriever computational graph retrieval process.
    """
    base_path = "test/fixtures/regression_tests"
    x = chainer.Variable(np.load(base_path + '/sample_input.npy'))

    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean = np.broadcast_to(mean.reshape((1, 3, 1, 1)), (1, 3, 224, 224))

    for arch in [L.GoogLeNet, L.VGG16Layers, L.ResNet50Layers]:
        c = "{}/{}_int8_calib_cache.dat".format(base_path, arch.__name__)

        @generator(errs=(0.0001, 0.005, 0.03),
                   batch_sizes=(1, 8), int8_calib_cache=c,
                   name_suffix='_{}'.format(arch.__name__))
        def regression_imagenet():
            net = arch()
            y = net(x.transpose(0, 3, 1, 2) - mean)['prob']
            return {'input': x}, {'prob': y.reshape(1, 1000, 1, 1)}


###############################################################################


def atos(ar, in_int=False):
    if type(ar) is chainer.Variable:
        ar = ar.data
    if in_int:
        ar = ar.astype(int)
    return ",".join([str(t) for t in ar.flatten()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-f", type=str, required=False,
                        default="*", help="Wildcard filter")
    args = parser.parse_args()

    test_cases = []
    for gen_info in generators:
        gen, errs, batch_sizes, name_suffix, \
                int8_calib_cache, custom_dump_functions = gen_info
        case_name = gen.__name__ + name_suffix
        if not fnmatch.fnmatch(case_name, args.filter):
            continue

        out = 'test/fixtures/model/' + case_name
        retriever = chainer_trt.ModelRetriever(out)
        for fun, dump in custom_dump_functions.items():
            retriever.add_dump_function(fun, dump)
        with chainer.using_config('train', False):
            with chainer_trt.RetainHook():
                inputs, outputs = gen()

        # Let ModelRetriever know what is the input
        input_csv_names = []
        for name, x in inputs.items():
            fn = name + '.csv'
            input_csv_names.append(fn)
            open(out + '/' + fn, "wt").write(atos(x))
            retriever.register_inputs(x, name=name)

        # Let ModelRetriever retrieve from outputs
        output_csv_names = []
        output_shapes = []
        for name, y in outputs.items():
            fn = name + '.csv'
            output_csv_names.append(fn)
            retriever(y, name=name)
            open(out + '/' + fn, "wt").write(atos(y))
            output_shapes.append(y.shape[1:])

        # Save dumped test case network (model.json)
        retriever.save()

        # Generate test case definition
        for bs in batch_sizes:
            for dtype, error in zip(['kFLOAT', 'kHALF', 'kINT8'], errs):
                test_cases.append({
                    "fixture": case_name, "inputs": input_csv_names,
                    "expected_outputs": output_csv_names,
                    "output_dims": output_shapes,
                    "batch_size": bs, "dtype": dtype,
                    "acceptable_absolute_error": error,
                    "external_plugins": [f.__name__ for f
                                         in custom_dump_functions.keys()],

                    # This is only used for int8 mode
                    "int8_calib_cache": int8_calib_cache
                })
                print("Generated {} (batch={}, dtype={}, err={})"
                      .format(case_name, bs, dtype, error))

    # Save test case definitions
    dst = 'test/fixtures/model_fixtures.json'
    with open(dst, 'wt') as f:
        json.dump(test_cases, f, indent=2)
    print("Saved to", dst)


if __name__ == '__main__':
    main()
