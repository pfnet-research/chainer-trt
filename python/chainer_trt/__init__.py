# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

from chainer_trt import _version                           # NOQA
from chainer_trt import functions                          # NOQA
from chainer_trt import json_encoder                       # NOQA

from chainer_trt.model_retriever import MarkPrefixHook     # NOQA
from chainer_trt.model_retriever import ModelRetriever     # NOQA
from chainer_trt.model_retriever import RetainHook         # NOQA
from chainer_trt.model_retriever import TracebackHook      # NOQA

from chainer_trt.dla_support import DLA                    # NOQA
from chainer_trt.dla_support import DLABlock               # NOQA

try:
    from chainer_trt import python_interface               # NOQA
    from chainer_trt.python_interface import Buffer        # NOQA
    from chainer_trt.python_interface import Infer         # NOQA
    is_python_interface_built = True                            # NOQA
except Exception:
    # User can confirm if python interface is correctly built
    # by checking `chainer_trt.is_python_interface_built`
    is_python_interface_built = False                           # NOQA

    from chainer_trt import dummy_python_interface         # NOQA
    from chainer_trt.dummy_python_interface import Buffer  # NOQA
    from chainer_trt.dummy_python_interface import Infer   # NOQA

__version__ = _version.__version__                              # NOQA
