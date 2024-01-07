from typing import Optional
import numpy as np

from ._core import enforce_metrics_invariants


def kendall(
    x: np.ndarray | list, y: Optional[np.ndarray | list] = None,
    weights: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None,
    normalize: Optional[bool] = False
) -> float:
    """
        Computes the Generalized Kendall-Tau distance between rankings [1].

        [1] Kumar and Vassilvitskii. “Generalized Distances between Rankings.”
            Proceedings of the 19th International Conference on World Wide Web,
            ACM, 2010, https://doi.org/10.1145/1772690.1772749.

    :param x: _description_
    :type x: np.ndarray
    :param y: _description_
    :type y: np.ndarray
    :param weights: _description_, defaults to None
    :type weights: Optional[np.ndarray], optional
    :param distance: _description_, defaults to None
    :type distance: Optional[np.ndarray], optional
    :param normalize: _description_, defaults to False
    :type normalize: Optional[bool], optional
    """

    x, y = enforce_metrics_invariants(x, y)

    inversions = 0
    n = x.shape[-1]
    for i in range(n):
        for j in range(i+1, n):
            inversion = x[i] < x[j] and y[i] > y[j]
            inversion |= x[i] > x[j] and y[i] < y[j]

            inversions += 1 if inversion else 0

    if normalize:
        inversions /= n * (n - 1) / 2

    return inversions


def fast_kendall(
    x: np.ndarray, y: Optional[np.ndarray | list] = None,
    weights: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None,
    normalize: Optional[bool] = False
) -> float:
    """
        Computes the Kendal-Tau distance between rankings using a reduction to
        sorting with merge sort. This implementation has a time complexity
        on the order of O(n log n).

    :param x: _description_
    :type x: np.ndarray
    :param y: _description_
    :type y: Optional[np.ndarray | list]
    :param weights: _description_, defaults to None
    :type weights: Optional[np.ndarray], optional
    :param distance: _description_, defaults to None
    :type distance: Optional[np.ndarray], optional
    :param normalize: _description_, defaults to False
    :type normalize: Optional[bool], optional
    :return: _description_
    :rtype: int
    """

    def _merge_sort():
        pass

    raise NotImplementedError()
