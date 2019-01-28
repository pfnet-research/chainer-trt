# Copyright (c) 2019 Preferred Networks, Inc. All rights reserved.

import chainer


def DLA(fn):
    """Tell chainer-trt to enable DLA for a function

    By using DLA, you can tell chainer-trt and TensorRT to try to use DLA
    for the function call.

    ```
    h = DLA(F.f1(x))
    ```

    This doesn't guarantee all the layers to run on DLA.
    """
    fn.creator._chainer_trt_enable_dla = True
    return fn


class DLABlock(chainer.FunctionHook):
    """Tell chainer-trt to enable DLA within a block

    By surrounding a certain series of Chainer Function call with DLABlock,
    you can tell chainer-trt and TensorRT to try to use DLA for the block.

    ```
    with DLABlock():
        h = F.f1(x)
        h = F.f2(h)
    ```

    This doesn't guarantee all the layers to run on DLA.
    """
    name = 'DLABlock'

    def forward_postprocess(self, function, _):
        function._chainer_trt_enable_dla = True
