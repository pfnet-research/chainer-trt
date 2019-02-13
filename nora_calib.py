#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import shutil

import chainer
import numpy as np
import tqdm

import cv2
import chainer_trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_list')
    parser.add_argument('dest')
    args = parser.parse_args()

    net = chainer.links.ResNet50Layers()

    min_max = dict()
    for img in tqdm.tqdm(open(args.calib_list).read().splitlines()):
        x = cv2.imread(img)
        x = cv2.resize(x, (224, 224))[None, :].astype(np.float32)
        x = chainer.Variable(x)

        retriever = chainer_trt.ModelRetriever(args.dest, verbose=True)
        retriever.register_inputs(x, name="input")
        mean = np.array([103.939, 116.779, 123.68])
        mean = mean.reshape((1, 3, 1, 1)).astype(np.float32)
        mean = np.broadcast_to(mean, (1, 3, 224, 224))
        with chainer.using_config('train', False):
            with chainer_trt.RetainHook():
                x = x.transpose((0, 3, 1, 2))   # hwc2chw
                x = x - mean
                y = net(x)['prob']
                retriever(y, name='prob')
                retriever.save()

                for f in tqdm.tqdm(glob.glob(args.dest + '/*output.tensor')):
                    name = os.path.basename(f).replace('_output.tensor', '')
                    if name not in min_max:
                        min_max[name] = (np.finfo(np.float32).max,
                                         np.finfo(np.float32).min)

                    b = open(f, 'rb')
                    b.readline(), b.readline()  # header
                    buf = b.read()
                    x = np.frombuffer(buf, dtype=np.float32)
                    _min, _max = min_max[name]
                    min_max[name] = (min(_min, min(x)), max(_max, max(x)))
        print(min_max)
    print(min_max)

    model_json = args.dest + '/model.json'
    shutil.copyfile(model_json, model_json + '.bkf')
    model = json.load(open(model_json))

    for layer, rng in min_max.items():
        r = max(abs(rng[0]), abs(rng[1]))
        for l in model['layers']:
            if l['name'] == layer:
                l['value_range'] = float(r)
                break
    json.dump(model, open(model_json, "wt"), indent=2)


if __name__ == "__main__":
    main()
