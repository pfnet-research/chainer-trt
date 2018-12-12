# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.


def require_libpyrt(tester_func):
    """A decorator to enable test cases only when libpyrt can be imported"""
    def nothing(*args, **kwargs):
        pass

    def run_test(*args, **kwargs):
        tester_func(*args, **kwargs)

    try:
        from libpyrt import Buffer  # NOQA
        from libpyrt import Infer   # NOQA
    except ImportError:
        return nothing
    return run_test
