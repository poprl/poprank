from typing import Any, List
from popcore import Interaction, Population

from ..core import RateModule, Rate
from ..functional.rates import bayeselo, EloRate


class Bayeselo(RateModule[EloRate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1, elo_base: float = 10.0,
        elo_spread: float = 400.0, elo_draw: float = 97.3,
        elo_advantage: float = 32.8, iterations: int = 1000,
        tolerance: float = 1e-5
    ):
        super().__init__(population, rates, default_rate)
        self._elo_base = elo_base
        self._elo_spread = elo_spread
        self._elo_draw = elo_draw
        self._elo_advantage = elo_advantage
        self._iterations = iterations
        self._tolerance = tolerance

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[EloRate]:
        return bayeselo(
            self._population.players, interactions, self.rates,
            elo_base=kwds.get('elo_base', self._elo_base),
            elo_spread=kwds.get('elo_spread', self._elo_spread),
            elo_draw=kwds.get('elo_draw', self._elo_draw),
            elo_advantage=kwds.get('elo_advantage', self._elo_advantage),
            iterations=kwds.get('iterations', self._iterations),
            tolerance=kwds.get('tolerance', self._tolerance)
        )
