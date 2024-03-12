from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Metric(ABC):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def _compute(
        self, x: np.ndarray, y: np.ndarray, *args, **kwargs
    ) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _max(self, n: int) -> int:
        raise NotImplementedError()

    def __call__(
        self, x: np.ndarray, y: np.ndarray, *args: Any, **kwds: Any
    ) -> float:
        return self._compute(x, y, *args, **kwds)

    def max(self, n: int) -> int:
        """
            Computes the upper bound value of the metric
            on the space of n-ranked permutations.

        :param n: number of alternatives.
        :type n: int
        :return: metric upper bound.
        :rtype: int
        """
        return self._max(n)
