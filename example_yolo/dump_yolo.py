import argparse
import numpy as np

import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv2

import chainer_trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='voc0712')
    parser.add_argument('--out', '-o', type=str, default='yolo_dump')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode')
    args = parser.parse_args()
    model = YOLOv2(n_fg_class=len(voc_bbox_label_names),
                   pretrained_model=args.pretrained_model)

    x = chainer.Variable(np.random.random((1, 3, 416, 416)).astype(np.float32))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        x.to_gpu()

    retriever = chainer_trt.ModelRetriever(args.out, verbose=args.verbose)
    retriever.register_inputs(x, 'x')
    with chainer.using_config('train', False), chainer_trt.RetainHook():
        locs, objs, confs = model(x)

    retriever(locs, name='locs')
    retriever(objs, name='objs')
    retriever(confs, name='confs')
    retriever.save()


if __name__ == '__main__':
    main()
