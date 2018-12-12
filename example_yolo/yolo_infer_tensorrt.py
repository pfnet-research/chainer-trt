import argparse
import matplotlib.pyplot as plt
import time

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv2
from chainercv import utils
from chainercv.visualizations import vis_bbox

import chainer_trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--engine', required=False,
                        help='TensorRT engine file. If None, chainer-based '
                             'inference will run')
    parser.add_argument('--n-try', '-n', type=int, default=1000)
    parser.add_argument('--chainer', action='store_true')
    parser.add_argument('image')
    args = parser.parse_args()

    model = YOLOv2(n_fg_class=len(voc_bbox_label_names),
                   pretrained_model='voc0712')
    img = utils.read_image(args.image, color=True)

    chainer.cuda.get_device_from_id(args.gpu).use()

    if args.engine is not None:
        # Key idea:
        # `predict` method applies some pre-processings and preparations,
        # runs forward-pass and some post-processings.
        # We want to re-use all of them without copy-and-pasting the code,
        # while only forward-pass runs with TensorRT.
        # solution: replace __call__ and let `predict` call it,
        # as if it is the original python based forward pass
        def run_infer(self, x):
            return tuple(chainer.Variable(t) for t in infer(x))

        infer = chainer_trt.Infer(args.engine)     # NOQA
        type(model).__call__ = run_infer
        print("Loaded TensorRT inference engine {}".format(args.engine))
    else:
        print("Run Chainer based inference")

    # Run inference once and get detections
    # (1st execution tends to be slow in CUDA)
    bboxes, labels, scores = model.predict([img])

    before = time.time()
    for _ in range(args.n_try):
        model.predict([img])
    after = time.time()
    print('{:.3f}ms/img'.format(1000 * (after - before) / args.n_try))

    bbox, label, score = bboxes[0], labels[0], scores[0]
    vis_bbox(img, bbox, label, score, label_names=voc_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    main()
