from typing import Any, List
from popcore import Interaction, Population

from .._core import RateModule
from ..functional.rates import (
    multidim_elo, MultidimEloRate,
    bipartite_multidim_elo
)


class MultidimElo(RateModule[MultidimEloRate]):

    def __init__(
        self, population: Population, rates: List[MultidimEloRate] = None,
        default_rate: float = 1, k: float = 1, lr1: float = 16.0,
        lr2: float = 1.0, iterations: int = 100
    ):
        super().__init__(population, rates, default_rate)
        self._k = k
        self._lr1 = lr1
        self._lr2 = lr2
        self._iterations = iterations

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[MultidimEloRate]:
        return multidim_elo(
            self._population.players, interactions, self.rates,
            k=kwds.get('k', self._k), lr1=kwds.get('lr1', self._lr1),
            lr2=kwds.get('lr2', self._lr2),
            iterations=kwds.get('iterations', self._iterations)
        )


class BipartiteMultidimElo(RateModule[MultidimEloRate]):

    def __init__(
        self,
        population: Population, opponents: Population,
        rates: List[MultidimEloRate] = None,
        opponents_rates: List[MultidimEloRate] = None,
        default_rate: float = 1.0, k: float = 1, lr1: float = 16.0,
        lr2: float = 1.0, iterations: int = 100
    ):
        super().__init__(population, rates, default_rate)
        self._opponents = opponents
        if not opponents_rates:
            opponents_rates = self._defaults(opponents, default_rate)
        self._opponents_rates = [opponents_rates]
        self._k = k
        self._lr1 = lr1
        self._lr2 = lr2
        self._iterations = iterations

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[MultidimEloRate]:
        population_rates, opponents_rates = bipartite_multidim_elo(
            self._population.players, interactions, self.rates,
            self._opponents.players, self._opponents_rates[-1],
            k=kwds.get('k', self._k), lr1=kwds.get('lr1', self._lr1),
            lr2=kwds.get('lr2', self._lr2),
            iterations=kwds.get('iterations', self._iterations)
        )
        self._opponents_rates.append(opponents_rates)
        return population_rates
