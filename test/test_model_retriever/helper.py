import os
import os.path
import shutil

import chainer

from chainer_trt import model_retriever


class ModelRetrieverHelper:
    dir = 'chainer_trt_model_retriever_test'

    def setup_method(self, _):
        self.retriever = model_retriever.ModelRetriever(self.dir)

    def teardown_method(self, _):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)

    # The "ModelRetrieverHelper.case" decorator has to be specified
    # when calling chainer functions
    @classmethod
    def case(cls, f):
        def call(*args, **kwargs):
            with chainer.using_config('train', False):
                with model_retriever.RetainHook():
                    f(*args, **kwargs)
        return call
