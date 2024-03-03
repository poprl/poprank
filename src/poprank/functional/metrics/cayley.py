from typing import Optional
import numpy as np

from .core import enforce_metrics_invariants


def cayley(
    x: np.ndarray, y: Optional[np.ndarray | list] = None,
) -> float:
    """_summary_

    :param x: _description_
    :type x: np.ndarray
    :param y: _description_, defaults to None
    :type y: Optional[np.ndarray  |  list], optional
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

    raise NotImplementedError()
