import numpy as np

from ..functional.metrics import kendall
from ._base import Metric


class Kendall(Metric):

    def __init__(self) -> None:
        super().__init__("kendall")

    def _compute(
        self, x: np.ndarray, y: np.ndarray, *args, **kwargs
    ) -> float:
        return kendall(x, y, *args, **kwargs)

    def _max(self, n: int) -> int:
        return n * (n - 1) / 2
