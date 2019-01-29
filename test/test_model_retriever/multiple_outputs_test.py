import chainer
import chainer.functions as F
import numpy as np
import json

from helper import ModelRetrieverHelper


def dump_split_axis(model_retriever, func, initial_params):
    x = func.inputs[0]
    source = model_retriever.get_source_name(x)
    initial_params['source'] = source
    initial_params['axis'] = func.axis
    initial_params['sections'] = func.sections

    weights = None  # No weights
    return initial_params, weights


class TestMultipleOutput(ModelRetrieverHelper):

    @ModelRetrieverHelper.case
    def test_dump_split_axis(self):
        x = np.random.rand(3, 10, 10).astype(np.float32)
        x = chainer.Variable(x)

        self.retriever.add_dump_function(F.array.split_axis.SplitAxis,
                                         dump_split_axis)

        y1, y2, y3 = F.split_axis(x, 3, axis=0)
        self.retriever(y1, name='y1')
        self.retriever(y2, name='y2')
        self.retriever(y3, name='y3')
        j = self.retriever.generate_json_source()

        # Only one SplitAxis layer should be found
        assert len([l for l in j['layers'] if l['type'] == 'SplitAxis']) == 1

        # NN should have 3 outputs
        assert len(j['outputs']) == 3
