import chainer
import numpy as np

from helper import ModelRetrieverHelper


class TestInputs(ModelRetrieverHelper):

    @ModelRetrieverHelper.case
    def test_input_order(self):
        xs = [chainer.Variable(np.ones((1, 3, 32, 32))) for _ in range(5)]
        y = xs[0]
        for x in xs[1:]:
            y = y + x
        for i in range(len(xs)):
            self.retriever.register_inputs(xs[i], name='in_' + str(i))
        self.retriever(y)
        j = self.retriever.generate_json_source()
        assert j['inputs'] == ['in_0', 'in_1', 'in_2', 'in_3', 'in_4']

    @ModelRetrieverHelper.case
    def test_unused_registerd_input_should_not_appear(self):
        xs = [chainer.Variable(np.ones((1, 3, 32, 32))) for _ in range(5)]
        y = xs[0]
        for x in xs[1:-1]:  # The last one is intentionally ignored
            y = y + x
        for i in range(len(xs)):
            self.retriever.register_inputs(xs[i], name='in_' + str(i))

        self.retriever(y)
        j = self.retriever.generate_json_source()
        assert j['inputs'] == ['in_0', 'in_1', 'in_2', 'in_3']

    @ModelRetrieverHelper.case
    def test_constants_should_not_appear_in_inputs(self):
        x = chainer.Variable(np.ones((1, 3, 32, 32)))
        c = chainer.Variable(np.ones((1, 3, 32, 32))) * 3
        y = x + c
        self.retriever.register_inputs(x, name='in')
        self.retriever(y)
        j = self.retriever.generate_json_source()
        assert j['inputs'] == ['in']
        assert sum(l['type'] == 'ConstantInput' for l in j['layers']) == 1

    @ModelRetrieverHelper.case
    def test_input_shouldnt_be_treated_as_constant_if_nothing_registered(self):
        x1 = chainer.Variable(np.ones((1, 3, 32, 32)))
        x2 = chainer.Variable(np.ones((1, 3, 32, 32)) * 2)
        y = x1 + x2
        self.retriever(y)
        j = self.retriever.generate_json_source()
        assert len(j['inputs']) == 2
        assert sum(l['type'] == 'input' for l in j['layers']) == 2
        assert sum(l['type'] == 'ConstantInput' for l in j['layers']) == 0

    @ModelRetrieverHelper.case
    def test_out_in_middle_of_already_read_graph(self):
        x = chainer.Variable(np.zeros((1, 3, 32, 32)))
        y1 = x + 1
        y2 = y1 * 2
        self.retriever(y2, name='y2')
        self.retriever(y1, name='y1')
        j = self.retriever.generate_json_source()
        assert len(j['outputs']) == 2
        y2_id, y2_name = j['outputs'][0]
        y1_id, y1_name = j['outputs'][1]
        assert y2_id == 'MulConstant-1-1_0' and y2_name == 'y2'
        assert y1_id == 'AddConstant-0-1_0' and y1_name == 'y1'
