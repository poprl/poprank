from typing import Optional
import numpy as np

from ._core import _enforce_metrics_invariants


def lee(
    x: np.ndarray, y: Optional[np.ndarray | list] = None,
    weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    """
       Computes Lee's distance between two ranks. See [1].

       [1] Deza, M., and Tayuan H. “Metrics on Permutations, a Survey.”
        J. Comb. Inf. Sys. Sci., vol. 23, no. 1, Feb. 1997.

    :param x: _description_
    :type x: np.ndarray
    :param y: _description_
    :type y: np.ndarray
    :param weight: _description_, defaults to None
    :type weight: Optional[np.ndarray], optional
    :param distance: _description_, defaults to None
    :type distance: Optional[np.ndarray], optional
    :param normalize: _description_, defaults to False
    :type normalize: Optional[bool], optional
    :return: _description_
    :rtype: float
    """
    x, y = _enforce_metrics_invariants(x, y)

    d = np.abs(x - y)
    n = x.shape[-1]

    return np.sum(np.min(d, -d + n), axis=-1)
