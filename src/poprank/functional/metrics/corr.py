from typing import Optional
import numpy as np

from .core import enforce_metrics_invariants


def corr(
    x: np.ndarray, y: Optional[np.ndarray | list] = None,
) -> float:
    """
        Computes Spearman's rank correlation. See [1], Chapter 6B "Some
        Metrics on Permutations".

        [1] Diaconis, Persi. Group Representations in Probability
        and Statistics. Institute of Mathematical Statistics, 1988.

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
    x, y = enforce_metrics_invariants(x, y)

    return np.sum((x - y) ** 2)
