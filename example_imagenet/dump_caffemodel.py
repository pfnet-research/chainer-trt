# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import argparse

import chainer
from chainer.links import caffe
import numpy

import chainer_trt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write chainer model to TensorRT mid-expression file')
    parser.add_argument('-m', '--caffemodel', type=str, required=True,
                        help='Import model file')
    parser.add_argument('-r', '--dest', type=str, required=True,
                        help='Name of output directory')
    parser.add_argument('-o', '--output-layers', type=str,
                        nargs='+', required=True, help='Name of output layers')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode')

    parser.add_argument('-W', '--in-width', type=int, default=227,
                        help='Input image width')
    parser.add_argument('-H', '--in-height', type=int, default=227,
                        help='Input image height')
    parser.add_argument('-D', '--in-depth', type=int, default=3,
                        help='Input image depth in bytes')
    args = parser.parse_args()

    print('Loading network...')
    net = caffe.CaffeFunction(args.caffemodel)
    retriever = chainer_trt.ModelRetriever(args.dest, verbose=args.verbose)
    retriever.preprocess_caffemodel(net)

    print('Calling forward pass...')
    x = numpy.random.rand(1, args.in_depth, args.in_height, args.in_width)
    x = x.astype(numpy.float32) * 255.0  # batch==1
    with chainer.using_config('train', False):
        with chainer_trt.RetainHook():
            y, = net(inputs={'data': x}, outputs=args.output_layers)
            y = chainer.functions.softmax(y)

    retriever(y)
    retriever.save()
