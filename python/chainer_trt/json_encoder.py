# Copyright (c) 2018 Preferred Networks, Inc. All rights reserved.

import json
import numpy

from chainer import cuda


class JSONEncoderEX(json.JSONEncoder):
    """Inherit JSONEncoder class to serialize numpy/cupy array.

    Ref: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """     # NOQA
    def default(self, obj):
        """Custom `default` method for serializing numpy/cupy array.

        It returns a serializable object for ``obj``,
        or calls the base implementation (to raise a ``TypeError``).
        """     # NOQA
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            if obj.ndim <= 1 and obj.size == 1:
                return obj[0]
            else:
                return obj.tolist()
        elif isinstance(obj, cuda.ndarray):
            return cuda.to_cpu(obj).tolist()
        else:
            return super(JSONEncoderEX, self).default(obj)
