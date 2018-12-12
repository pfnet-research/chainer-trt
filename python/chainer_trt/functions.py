# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import chainer
import chainer.functions as F


class ResizeArgmax(chainer.function_node.FunctionNode):
    def __init__(self, output_shape, axis):
        self.resize = F.array.resize_images.ResizeImages(output_shape)
        self.argmax = F.math.minmax.ArgMax(axis)

    def forward(self, inputs):
        x = inputs
        x = self.resize.forward(x)
        x = self.argmax.forward(x)
        return x

    def backward(self, indexes, grad_outputs):
        return None,


def resize_argmax(x, output_shape, axis):
    return ResizeArgmax(output_shape, axis).apply((x,))[0]
