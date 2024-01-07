import numpy as np


def sigmoid(x: float) -> float:
    """
       Numerically stable implementation of sigmoid.


    :param x: _description_
    :type x: float
    :param base: _description_, defaults to e
    :type base: float, optional
    :param spread: _description_, defaults to 1.0
    :type spread: float, optional
    :return: _description_
    :rtype: float
    """
    return np.exp(-np.logaddexp(0, -x))
