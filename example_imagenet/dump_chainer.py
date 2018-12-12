# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import argparse

import chainer.links
import numpy

import chainer_trt

if __name__ == '__main__':
    model_choices = [
        'ResNet50Layers', 'ResNet101Layers', 'ResNet152Layers',
        'VGG16Layers', 'GoogLeNet'
    ]

    parser = argparse.ArgumentParser(
        description='Save chainer predefined models to TensorRT')
    parser.add_argument('model', type=str, choices=model_choices,
                        help='Import model file')
    parser.add_argument('-r', '--dest', type=str, required=True,
                        help='Name of output directory')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode')
    args = parser.parse_args()

    print('Loading network...')
    net = getattr(chainer.links, args.model)()

    x = numpy.random.random((1, 224, 224, 3)).astype(numpy.float32)
    x = chainer.Variable(x)

    retriever = chainer_trt.ModelRetriever(args.dest, verbose=args.verbose)
    retriever.register_inputs(x, name="input")

    print('Calling forward pass...')
    mean = numpy.array([103.939, 116.779, 123.68])
    mean = mean.reshape((1, 3, 1, 1)).astype(numpy.float32)
    mean = numpy.broadcast_to(mean, (1, 3, 224, 224))
    with chainer.using_config('train', False):
        with chainer_trt.RetainHook():
            x = x.transpose((0, 3, 1, 2))   # hwc2chw
            x = x - mean
            y = net(x)['prob']

    retriever(y, name="prob")
    retriever.save()
