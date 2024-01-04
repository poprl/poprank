from typing import Any, List
from popcore import Population, Interaction

from ..core import Rank, Rate, RatedPlayer, RateModule


def rank_with(
    population: Population[RatedPlayer], interactions: List[Interaction],
    rate: RateModule
) -> Rank:
    """
        Ranks a population of players after a serie of interactions.

    :param population: _description_
    :type population: Population[RatedPlayer]
    :param interactions: _description_
    :type interactions: List[Interaction]
    :param rate: _description_
    :type rate: RateModule
    :return: _description_
    :rtype: Rank
    """
    rate(population, interactions)
    return Rank(population, order="rate")


def rate_with(
    population: Population, interactions: List[Interaction],
    rate: RateModule
) -> List[Rate]:
    """
        Rates a population of players after a serie of interactions.

    :param population: _description_
    :type population: Population
    :param interactions: _description_
    :type interactions: List[Interaction]
    :param rate: _description_
    :type rate: RateModule
    :return: _description_
    :rtype: List[Rate]
    """
    return rate(population, interactions)
