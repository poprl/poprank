import numpy as np
from ..functional.metrics import hamming
from ._base import Metric


class Hamming(Metric):
    def _compute(
        self, x: np.ndarray, y: np.ndarray, *args, **kwargs
    ) -> float:
        return hamming(x, y, *args, **kwargs)

    def _max(self, n: int) -> int:
        return n
