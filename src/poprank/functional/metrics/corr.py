
def corr(
    x: np.ndarray, y: Optional[np.ndarray | list] = None,
    weight: Optional[np.ndarray] = None,
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
    x, y = _enforce_metrics_invariants(x, y)

    return np.sum((x - y) ** 2)