from typing import Optional
import numpy as np

from ...excepts import raise_with_message_code


def _enforce_metrics_invariants(x, y) -> tuple:
    x = np.asarray(x)

    if y is None:
        y = np.array(np.arange(x.shape[-1]) + 1)  # identity permutation
    else:
        y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    return x, y

