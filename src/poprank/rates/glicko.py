from typing import Any, List
from popcore import Interaction, Population

from .._core import RateModule, Rate
from ..functional.rates import (
    glicko, glicko2, GlickoRate, Glicko2Rate
)


class Glicko(RateModule[GlickoRate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1.0, base: float = 10.0, spread: float = 400.0,
        uncertainty_increase: float = 34.6, rating_dev_unrated: float = 350.0
    ):
        super().__init__(population, rates, default_rate)
        self._uncertainty_increase = uncertainty_increase
        self._rating_dev_unrated = rating_dev_unrated
        self._base = base
        self._spread = spread

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[GlickoRate]:
        return glicko(
            self._population.players, interactions, self.rates,
            uncertainty_increase=kwds.get(
                'uncertainty_increase', self._uncertainty_increase),
            rating_deviation_unrated=kwds.get(
                'rating_deviation_unrated', self._rating_dev_unrated),
            base=kwds.get('base', self._base),
            spread=kwds.get('spread', self._spread)
        )


class Glicko2(RateModule[Glicko2Rate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1.0, rating_dev_unrated: float = 350.0,
        volatility_constraint: float = 0.5, epsilon: float = 1e-6,
        unrated_player_rate: float = 1500, conversion_std: float = 173.7178

    ):
        super().__init__(population, rates, default_rate)
        self._rating_dev_unrated = rating_dev_unrated
        self._volatility_constraint = volatility_constraint
        self._epsilon = epsilon
        self._unrated_player_rate = unrated_player_rate
        self._conversion_std = conversion_std

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[Glicko2Rate]:
        return glicko2(
            self._population.players, interactions, self.rates,
            rating_deviation_unrated=kwds.get(
                'rating_deviation_unrated', self._rating_dev_unrated),
            volatility_constraint=kwds.get(
                'volatility_constraint', self._volatility_constraint),
            epsilon=kwds.get('epsilon', self._epsilon),
            unrated_player_rate=kwds.get(
                'unrated_player_rate', self._unrated_player_rate),
            conversion_std=kwds.get('conversion_std', self._conversion_std)
        )
