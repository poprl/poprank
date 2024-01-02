from typing import Optional
import numpy as np

from .excepts import raise_with_message_code


def kendall(
    x: np.ndarray | list, y: np.ndarray | list, weights: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
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
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[-1] == y.shape[-1]

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
    x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    """
        Computes the Kendal-Tau distance between rankings using a reduction to
        sorting with merge sort. This implementation has a time complexity 
        on the order of O(n log n).

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
    :return: _description_
    :rtype: int
    """

    def _merge_sort():
        pass

    raise_with_message_code(
        "not_implemented", NotImplementedError, "fast_kendall")


def footrule(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    """
        Computes Spearman's footrule. Equivalent to the 1-norm between the two
        ranks. See [1], Chapter 6B and [2].

        [1] Diaconis, P. "Group Representations in Probability and Statistics".
        Institute of Mathematical Statistics, 1988.
        [2] Deza, M., and Tayuan H. “Metrics on Permutations, a Survey.”
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
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[-1] == y.shape[-1]

    return np.sum(np.abs(x - y))


def max(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    """
        Max norm betweem two ranks. See [1].

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
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[-1] == y.shape[-1]

    return np.max(np.abs(x - y))


def corr(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
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
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    return np.sum((x - y) ** 2)


def hamming(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    """
        Computes Hamming's distance. See [1], Chapter 6B "Some
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

    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    return np.sum((x != y).astype(np.int32), axis=-1)


def lee(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
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
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    d = np.abs(x - y)
    n = x.shape[-1]

    return np.sum(np.min(d, -d + n), axis=-1)


def cayley(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    x = np.asarray(x)
    y = np.asarray(y)

    assert x.shape[-1] == y.shape[-1]

    raise_with_message_code(
        "not_implemented", NotImplementedError, "cayley")


def ulam(
    x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
    distance: Optional[np.ndarray] = None, normalize: Optional[bool] = False
) -> float:
    pass
