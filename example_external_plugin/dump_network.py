import argparse

import chainer
import chainer_trt
import numpy as np


# A tiny Function definition
class Increment(chainer.FunctionNode):
    def __init__(self):
        pass

    def forward(self, x):
        return x[0] + 1,


def increment(x):
    return Increment().apply(x)[0]


def dump_increment(model_retriever, func, initial_params):
    x = func.inputs[0]
    source = model_retriever.get_source_name(x)

    # Insert arbitrary metadata about the node.
    # At least information of source layer would be necessary
    initial_params['source'] = source

    # If the layer has parameters, make a dict (str: {numpy,cupy}.array).
    # ModelRetriever automatically saves them to destination directory,
    # and embed relative path to them in parameter dictionary
    weights = None

    return initial_params, weights


class Net(chainer.Chain):
    def __init__(self):
        pass

    def __call__(self, x):
        return increment(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", required=True, type=str, help="Output path")
    args = parser.parse_args()

    x = chainer.Variable(np.random.random((1, 3, 2, 2)).astype(np.float32))
    retriever = chainer_trt.ModelRetriever(args.out)
    retriever.register_inputs(x, name="input")
    retriever.add_dump_function(Increment, dump_increment)

    # Run forward pass
    with chainer.using_config('train', False), chainer_trt.RetainHook():
        net = Net()
    y = net(x)

    retriever(y, name="output")
    retriever.save()
