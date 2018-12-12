# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import argparse
import time

import cupy
import cv2
import numpy

import chainer_trt


if hasattr(time, 'perf_counter'):
    timer = time.perf_counter   # py3
else:
    timer = time.time   # py2

mode = [
    'cupy',
    'numpy',
    'buffer'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run ImageNet TensorRT inference engine from Python')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='TensorRT file')
    parser.add_argument('-i', '--input-image', type=str, required=True,
                        help='Input image')
    parser.add_argument('-l', '--label', type=str,
                        default='example_imagenet/labels.txt',
                        help='ImageNet label file')
    parser.add_argument('-n', '--n-try', type=int, default=1,
                        help='How many times you try to run inference')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--with-name', action='store_true',
                        help='Specify if you use "named input/output" API of '
                             'chaniner-tensorrt')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--mode', default=mode[0], choices=mode,
                        help='Where the source data is.')
    args = parser.parse_args()

    print('Batch size = {}'.format(args.batch_size))

    print('Loading model')
    infer = chainer_trt.Infer(args.model)

    print('Loading labels')
    labels = open(args.label).read().splitlines()

    print('Loading image')
    x = cv2.imread(args.input_image)
    x = cv2.resize(x, (224, 224))
    x = x.astype(numpy.float32)
    if args.mode == 'cupy':
        print('Mode: directly feed cupy array')
        x = cupy.array(x)
    elif args.mode == 'numpy':
        print('Mode: feed numpy array')
        pass
    elif args.mode == 'buffer':
        print('Mode: use chainer_trt.Buffer')
        buf = chainer_trt.Buffer(infer, args.batch_size)
        if args.with_name:
            buf.input_to_gpu({'input': x})
        else:
            buf.input_to_gpu([x])

    print('Inference')
    before = timer()
    for i in range(args.n_try):
        if args.mode == 'buffer':
            infer(buf)
        else:
            if args.with_name:
                ys = infer({'input': x})
            else:
                ys = infer([x])
    cupy.cuda.Stream.null.synchronize()
    avg_tm = timer() - before

    if args.mode == 'cupy':
        y = ys['prob'] if args.with_name else ys[0]
        y = y.get()
    elif args.mode == 'numpy':
        y = ys['prob'] if args.with_name else ys[0]
    elif args.mode == 'buffer':
        y = numpy.zeros((args.batch_size * 1000,), numpy.float32)
        if args.with_name:
            buf.output_to_cpu({'prob': y})
        else:
            buf.output_to_cpu([y])

    avg_tm /= args.n_try
    if args.mode in ('cupy', 'buffer'):
        print('Average inference time (not including CPU->GPU transfer) = {}ms'
              .format(1000 * avg_tm))
    else:
        print('Average inference time (including CPU->GPU transfer) = {}ms'
              .format(1000 * avg_tm))

    scores = list(zip(y.flatten()[:1000], labels))
    scores.sort()
    scores.reverse()
    for score, label in scores[:5]:
        print('{:.6f} - {}'.format(score, label))
