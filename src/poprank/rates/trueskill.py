from typing import Any, List
from popcore import Interaction, Population
from popcore.core import Interaction

from ..core import RateModule, Rate
from ..functional.rates import trueskill, TrueSkillRate


class TrueSkill(RateModule[TrueSkillRate]):

    def __init__(
        self, population: Population, rates: List[Rate] = None, default_rate: float = 1):
        super().__init__(population, rates, default_rate)

    def _rate(self, interactions: List[Interaction], **kwds: Any) -> List[TrueSkillRate]:
        return super()._rate(interactions, **kwds)