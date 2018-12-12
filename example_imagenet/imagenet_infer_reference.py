# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import argparse
import time

import cv2

import chainer
import chainer.links
import chainer.links.caffe
import numpy


if hasattr(time, 'perf_counter'):
    timer = time.perf_counter   # py3
else:
    timer = time.time   # py2


def infer(net, x_gpu, output):
    if type(net) is chainer.links.caffe.CaffeFunction:
        y_gpu, = net(inputs={'data': x_gpu}, outputs=[output])
    else:
        y_gpu = net(x_gpu)[output]
    return y_gpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write chainer model to TensorRT mid-expression file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Import model file')
    parser.add_argument('-i', '--input-image', type=str, required=True,
                        help='Input image')
    parser.add_argument('-l', '--label', type=str,
                        default='example_imagenet/labels.txt',
                        help='ImageNet label file')
    parser.add_argument('-o', '--out', type=str, default='prob',
                        help='Name of output layer')
    parser.add_argument('-n', '--n-try', type=int, default=1,
                        help='How many times you try to run inference')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size')
    args = parser.parse_args()

    print('Loading model')
    model_choices = [
        'ResNet50Layers', 'ResNet101Layers', 'ResNet152Layers',
        'VGG16Layers', 'GoogLeNet'
    ]
    if args.model in model_choices:
        net = getattr(chainer.links, args.model)()
    else:
        net = chainer.links.caffe.CaffeFunction(args.model)
    net.to_gpu(args.gpu)

    print('Loading labels')
    labels = open(args.label).read().splitlines()

    print('Loading image')
    mean = numpy.array([103.939, 116.779, 123.68], dtype=numpy.float32)
    mean = mean.reshape((3, 1, 1))
    x = cv2.imread(args.input_image)
    x = cv2.resize(x, (224, 224))
    x = x.transpose(2, 0, 1).astype(numpy.float32) - mean
    x = numpy.stack([x] * args.batch_size)
    x_gpu = chainer.Variable(x)
    x_gpu.to_gpu(args.gpu)

    # Dummy inference
    # (First operation takes a few time for CUDA kernel generation,
    # that makes time measurement very inaccurate)
    infer(net, x_gpu, args.out)

    print('Inference')
    with chainer.using_config('train', False):
        before = timer()
        for i in range(args.n_try):
            y = infer(net, x_gpu, args.out)
        avg_tm = timer() - before

    avg_tm /= args.n_try
    print('Average inference time = {}ms'.format(1000 * avg_tm))

    y.to_cpu()
    scores = list(zip(y.data[0], labels))
    scores.sort()
    scores.reverse()
    for score, label in scores[:5]:
        print('{:.6f} - {}'.format(score, label))
