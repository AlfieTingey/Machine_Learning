import os
import numpy as np


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def delete_file(fname):
    try:
        os.remove("{}".format(fname))
    except Exception:
        pass

    try:
        os.remove("testing/{}".format(fname))
    except Exception:
        pass
