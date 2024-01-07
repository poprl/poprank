import numpy as np


def enforce_metrics_invariants(x, y) -> tuple:
    x = np.asarray(x)

    if y is None:
        y = np.array(np.arange(x.shape[-1]) + 1)  # Identity
    else:
        y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    return x, y
