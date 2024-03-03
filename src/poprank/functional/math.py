import math


def sigmoid(x: float, base: float = math.e) -> float:
    """
       Numerically stable implementation of sigmoid.


    :param x: value to compute the sigmoidal function
    :type x: float
    :param base: the base of the `sigmoid`, defaults to e
    :type base: float, optional
    :return: function value
    :rtype: float
    """
    if x >= 0.0:
        return 1.0 / (1 + pow(base, -x))
    return pow(base, x) / (1 + pow(base, x))
