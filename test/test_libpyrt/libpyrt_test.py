# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import cupy
import numpy as np
import pytest

from libpyrt_test_helper import require_libpyrt


@require_libpyrt
def test_infer_using_arrays():
    import chainer_trt
    fixture = 'test/fixtures/chainer_trt/raw_binding'
    rt = chainer_trt.Infer.build(fixture)

    for xp in (np, cupy):
        x1 = xp.random.random((1, 3, 8, 8)).astype(xp.float32)
        x2 = xp.random.random((1, 3, 8, 8)).astype(xp.float32)
        expected_out = (x1 + x2) * 2

        # inference from list of arrays
        ys = rt([x1, x2])
        y = ys[0]
        assert ((ys[0] - expected_out) ** 2).sum() / x1.size < 1e-6

        # inference from arrays with *args
        ys = rt(x1, x2)
        y = ys[0]
        assert ((ys[0] - expected_out) ** 2).sum() / x1.size < 1e-6

        # inference from dict of arrays
        ys = rt({'x1': x1, 'x2': x2})
        y = ys['out']
        assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6

        # inference from arrays with **kwargs
        ys = rt(x1=x1, x2=x2)
        y = ys['out']
        assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6


@require_libpyrt
def test_infer_using_arrays_error_case():
    import chainer_trt
    fixture = 'test/fixtures/chainer_trt/raw_binding'
    rt = chainer_trt.Infer.build(fixture)

    for xp in (np, cupy):
        x1 = xp.random.random((1, 3, 8, 8)).astype(xp.float32)
        x2 = xp.random.random((1, 3, 8, 8)).astype(xp.float32)
        x3 = xp.random.random((1, 3, 8, 8)).astype(xp.float32)
        expected_out = (x1 + x2) * 2

        # insufficient
        with pytest.raises(ValueError):
            rt({'x1': x1})
        with pytest.raises(ValueError):
            rt(x1=x1)
        try:
            rt([x1])
            pytest.fail("It should raise an error for input insufficiency")
        except Exception:
            pass

        # too much (when passing a list)
        try:
            rt([x1, x2, x3])
            pytest.fail("It should raise an error for input insufficiency")
        except Exception:
            pass

        # *args and **kwargs are specified
        with pytest.raises(ValueError):
            rt({'x1': x1}, x2=x2)

        # nothing is specified
        with pytest.raises(ValueError):
            rt()

        # unknown type is specified
        with pytest.raises(ValueError):
            rt('hello world')

        # when using named inputs, allow too much inputs (simply ignore)
        y = rt({'x1': x1, 'x2': x2, 'x3': x3})['out']
        assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6
        y = rt(x1=x1, x2=x2, x3=x3)['out']
        assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6


@require_libpyrt
def test_infer_using_buffer():
    import chainer_trt
    fixture = 'test/fixtures/chainer_trt/raw_binding'
    rt = chainer_trt.Infer.build(fixture)
    buf = rt.create_buffer(1)

    x1 = np.random.random((1, 3, 8, 8)).astype(np.float32)
    x2 = np.random.random((1, 3, 8, 8)).astype(np.float32)
    expected_out = (x1 + x2) * 2

    # infer using list of arrays
    y = np.zeros_like(x1)
    buf.input_to_gpu([x1, x2])
    rt(buf)
    buf.output_to_cpu([y])
    assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6

    # infer using dict of arrays
    y = np.zeros_like(x1)
    buf.input_to_gpu({'x1': x1, 'x2': x2})
    rt(buf)
    buf.output_to_cpu({'out': y})
    assert ((y - expected_out) ** 2).sum() / x1.size < 1e-6


@require_libpyrt
def test_infer_using_buffer_error_case():
    import chainer_trt
    fixture = 'test/fixtures/chainer_trt/raw_binding'
    rt = chainer_trt.Infer.build(fixture)
    buf = rt.create_buffer(1)

    x1 = np.random.random((1, 3, 8, 8)).astype(np.float32)
    x2 = np.random.random((1, 3, 8, 8)).astype(np.float32)
    x3 = np.random.random((1, 3, 8, 8)).astype(np.float32)

    # insufficient
    with pytest.raises(ValueError):
        buf.input_to_gpu([x1])
    with pytest.raises(ValueError):
        buf.input_to_gpu({'x1': x1})
    with pytest.raises(ValueError):
        buf.output_to_cpu(dict())

    # too much (when passing a list)
    with pytest.raises(ValueError):
        buf.input_to_gpu([x1, x2, x3])

    # when using named inputs, allow too much inputs (simply ignore)
    buf.input_to_gpu({'x1': x1, 'x2': x2, 'x3': x3})
