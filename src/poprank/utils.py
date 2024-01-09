from typing import List, Optional
import numpy as np

from ._core import Interaction, Population


def to_pairwise(interactions: List[Interaction]) -> List[Interaction]:
    """
       Converts a list of interactions of any order into a list
       of pairwise (order 2) interactions.

    :param interactions: list of interactions to convert
    :type interactions: List[Interaction]
    :return: list of pairwise interactions.
    :rtype: List[Interaction]
    """
    pairwise = []
    for interaction in interactions:
        pairwise.extend(
            interaction.as_pairs()
        )
    return pairwise


def to_payoff_matrix(
    interactions: List[Interaction],
    population: Optional[Population] = None,
    reduction: Optional[str] = "sum"
) -> np.ndarray:
    """
        Reduces the list of interactions to a pairwise
        payoff matrix using a reduction strategy.

    :param reduce: strategy used to reduce multiple pairwise 
        interactions between the same players. "sum" or "avg".
        Defaults to "sum"
    :type reduce: str, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: np.ndarray
    """
    interactions = to_pairwise(interactions)

    if population is None:
        population = Population.from_players_interactions(interactions)

    payoffs = np.zeros(
        shape=(population.size, population.size), dtype=np.float32)

    for interaction in interactions:
        player = population[interaction.players[0]]
        opponent = population[interaction.players[1]]
        if reduction == "sum":
            payoffs[player, opponent] += interaction.outcomes[0]
            payoffs[opponent, player] += interaction.outcomes[1]
        elif reduction == "avg":
            raise NotImplementedError()
        else:
            raise ValueError()  # TODO: Execption handling.

    return payoffs


def to_win_matrix(
    interactions: List[Interaction],
    population: Optional[Population] = None,
    normalize: Optional[bool] = False
) -> np.ndarray:
    """
        Reduces the list of interactions to a win matrix. If
        normalize = True, the entries are pairwise win rates.

    :param interactions: _description_
    :type interactions: List[Interaction]
    :param population: _description_, defaults to None
    :type population: Optional[Population], optional
    :param normalize: _description_, defaults to False
    :type normalize: Optional[bool], optional
    :return: _description_
    :rtype: np.ndarray
    """
    interactions = to_pairwise(interactions)

    if population is None:
        population = Population.from_players_interactions(
            interactions
        )

    win_matrix = np.zeros(
        shape=(population.size, population.size), dtype=np.float32)

    for interaction in interactions:
        player = population[interaction.players[0]]
        opponent = population[interaction.players[1]]

        player_outcome, opponent_outcome = interaction.outcomes

        if player_outcome > opponent_outcome:
            win_matrix[player, opponent] += 1.0
        elif player_outcome < opponent_outcome:
            win_matrix[opponent, player] += 1.0

    if normalize:
        win_matrix /= win_matrix + win_matrix.T  # broadcasting.

    return win_matrix


def to_margin_matrix(
    interactions: List[Interaction],
    population: Optional[Population] = None,
) -> np.ndarray:
    """
        Computes S(x, y) = N(x, y) - N(y, x)

    :param population: _description_
    :type population: Population
    :param interactions: _description_
    :type interactions: List[Interaction]
    :param reduction: _description_, defaults to "sum"
    :type reduction: Optional[str], optional
    :return: _description_
    :rtype: np.ndarray
    """
    win_matrix = to_win_matrix(
        population, interactions
    )

    return win_matrix - win_matrix.T

# TODO: Implement
# def to_kemeny_matrix(
#     interactions: List[Interaction],
#     population: Optional[Population] = None,
#     reduction: Optional[str] = "none"
# ):
#     from math import comb
#     pass
