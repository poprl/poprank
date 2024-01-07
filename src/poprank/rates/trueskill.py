from typing import Any, List
from popcore import Interaction, Population

from ..core import RateModule, Rate
from ..functional.rates import trueskill, TrueSkillRate


class TrueSkill(RateModule[TrueSkillRate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1.0, dynamic_factor: float = 1/12,
        beta: float = 25/6, draw_probability: float = 1e-1,
        weights: list[list[float]] = None, iterations: int = 10, tolerance=1e-4
    ):
        super().__init__(population, rates, default_rate)
        self._dynamic_factor = dynamic_factor
        self._beta = beta
        self._draw_probability = draw_probability
        self._weights = weights
        self._iterations = iterations
        self._tolerance = tolerance

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[TrueSkillRate]:
        return trueskill(
            self._population.players, interactions, self.rates,
            dynamic_factor=kwds.get('dynamic_factor', self._dynamic_factor),
            beta=kwds.get('beta', self._beta),
            draw_probability=kwds.get(
                'draw_probability', self._draw_probability),
            weights=kwds.get('weights', self._weights),
            iterations=kwds.get('iterations', self._iterations),
            tolerance=kwds.get('tolerance', self._tolerance)
        )
