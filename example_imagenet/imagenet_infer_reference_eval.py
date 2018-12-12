# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import argparse
import math
import multiprocessing
import time

import cv2

import chainer
import chainer.links.caffe
import cupy
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


def each_slice(arr, n):
    return (arr[i:i + n] for i in range(0, len(arr), n))


def worker(prepare_event, start_event, model, gpu, base_path,
           worker_files, ret_n_tp_top1, ret_n_tp_top5):
    # preparation: load model
    model_choices = [
        'ResNet50Layers', 'ResNet101Layers', 'ResNet152Layers',
        'VGG16Layers', 'GoogLeNet'
    ]
    if model in model_choices:
        net = getattr(chainer.links, model)()
    else:
        net = chainer.links.caffe.CaffeFunction(model)

    # preparation: send to GPU
    xp = numpy
    if gpu != -1:
        chainer.cuda.get_device_from_id(gpu).use()
        net.to_gpu(gpu)
        xp = cupy

    # misc preparation
    mean = numpy.array([103.939, 116.779, 123.68], dtype=numpy.float32)
    mean = mean.reshape((3, 1, 1))

    # notify to the main process that the model has been prepared
    prepare_event.set()

    # wait for GO signal from main process
    start_event.wait()

    n_tp_top5, n_tp_top1 = 0, 0
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        for batch in worker_files:
            xs, gts = [], []
            for filename, label in batch:
                label = int(label)
                x = cv2.imread(base_path + '/' + filename)
                x = cv2.resize(x, (224, 224))
                x = xp.array(x.transpose(2, 0, 1).astype(numpy.float32) - mean)
                xs.append(x)
                gts.append(int(label))
            xs = xp.stack(xs)
            ys = chainer.cuda.to_cpu(infer(net, xs, args.out).data)
            ys = numpy.argsort(ys.data, axis=1)[:, ::-1]
            ys = ys[:, :5]  # extract top-5
            for y, gt in zip(ys, gts):
                if y[0] == gt:
                    n_tp_top1 += 1
                if gt in y:
                    n_tp_top5 += 1
    with ret_n_tp_top1.get_lock():
        ret_n_tp_top1.value += n_tp_top1
        ret_n_tp_top5.value += n_tp_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy of an ImageNet model')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Pretrained network name '
                             '(ResNet50Layers, etc...) or path to caffemodel')
    parser.add_argument('-i', '--image-list', type=str, required=True,
                        help='Path to ImageNet images with labels')
    parser.add_argument('-o', '--out', type=str, default='prob',
                        help='Name of output layer')
    parser.add_argument('-p', '--base-path', type=str, required=True,
                        help='Path to ImageNet images base directory')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n', '--n-workers', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    print('Loading image list')
    files = [line.split() for line in
             open(args.image_list, 'rt').read().splitlines()]
    n_files = len(files)

    # split equally to make lists for each worker
    files = numpy.array_split(files, args.n_workers)

    # split into batches for each worker
    files = [list(each_slice(worker_files, args.batch_size))
             for worker_files in files]

    print('Preparing workers')
    start_event = multiprocessing.Event()
    n_tp_top1 = multiprocessing.Value('i', 0)
    n_tp_top5 = multiprocessing.Value('i', 0)
    workers = []
    prepare_events = []
    for worker_files in files:
        pe = multiprocessing.Event()
        params = (pe, start_event, args.model, args.gpu, args.base_path,
                  worker_files, n_tp_top1, n_tp_top5)
        pr = multiprocessing.Process(target=worker, args=params)
        pr.start()
        workers.append(pr)
        prepare_events.append(pe)

    # wait until the preparation in each worker is completed
    for pe in prepare_events:
        pe.wait()
    print('Preparation completed')

    print('Starting inference')
    start_event.set()   # Send GO signal to each worker
    before = timer()
    for p in workers:
        p.join()
    tm = (timer() - before)
    n_batches = math.ceil(n_files / args.batch_size)
    batch_tm = tm / n_batches
    image_tm = tm / n_files
    print("Top1 accuracy = {}%".format(100.0 * n_tp_top1.value / n_files))
    print("Top5 accuracy = {}%".format(100.0 * n_tp_top5.value / n_files))
    print('Total time = {:.2f}s'.format(tm))
    print('Average time = {:.3f}ms/batch, {:.3f}ms/image'
          .format(batch_tm * 1000, image_tm * 1000))
