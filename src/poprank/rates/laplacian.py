from typing import Any, List
from popcore.core import Interaction, Population

from .._core import RateModule, Rate


class LaplacianRate(RateModule[Rate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1
    ):
        super().__init__(population, rates, default_rate)

    def _rate(
        self, interactions: List[Interaction], *args: Any, **kwds: Any) -> List[Rate]:
        return super()._rate(interactions, *args, **kwds)