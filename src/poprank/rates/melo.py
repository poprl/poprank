from typing import Any, List
from popcore import Interaction, Population

from ..core import RateModule
from ..functional.rates import multidim_elo, MultidimEloRate


class MultidimElo(RateModule[MultidimEloRate]):

    def __init__(
        self, population: Population, rates: List[MultidimEloRate] = None,
        default_rate: float = 1, k: float = 1, lr1: float = 16.0,
        lr2: float = 1.0
    ):
        super().__init__(population, rates, default_rate)
        self._k = k
        self._lr1 = lr1
        self._lr2 = lr2

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[MultidimEloRate]:
        return multidim_elo(
            self._population.players, interactions, self.rates,
            k=kwds.get('k', self._k), lr1=kwds.get('lr1', self._lr1),
            lr2=kwds.get('lr2', self._lr2)
        )
