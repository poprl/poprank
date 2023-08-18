from popcore import Interaction
from poprank.rates import EloRate

from ._bayeselo.data import (
    PopulationPairwiseStatistics
)
from ._bayeselo.core import BayesEloRating


def bayeselo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", elo_base: float = 10., elo_spread: float = 400.,
    elo_draw: float = 97.3, elo_advantage: float = 32.8,
    iterations: int = 10000, tolerance: float = 1e-5
) -> "list[EloRate]":
    """Rates players by calculating their new elo using a bayeselo approach

    Given a set of interactions and initial elo ratings, uses a
    Minorization-Maximization algorithm to estimate maximum-likelihood
    ratings.
    The Minorization-Maximization algorithm is performed for the number of
    specified iterations or until the changes are below the tolerance
    value, whichever comes first.
    Made to imitate https://www.remi-coulom.fr/Bayesian-Elo/

    Args:
        players (list[str]): The list of all players
        interactions (list[Interactions]): The list of all interactions
        elos (list[EloRate]): The initial ratings of the players
        elo_base (float, optional): The base of the exponent in the elo
            formula. Defaults to 10.0
        elo_spread (float, optional): The divisor of the exponent in the elo
            formula. Defaults to 400.0.
        elo_draw (float, optional): The probability of drawing.
            Defaults to 97.3.
        elo_advantage (float, optional): The home-field-advantage
            expressed as rating points. Defaults to 32.8.
        iterations (int, optional): The maximum number of iterations the
            Minorization-Maximization algorithm will go through.
            Defaults to 10000.
        tolerance (float, optional): The error threshold below which the
            Minorization-Maximization algorithm stopt. Defaults to 1e-5.

    Returns:
        list[EloRate]: The updated ratings of all players
    """

    # This check is necessary, otherwise the algorithm raises a
    # divide by 0 error
    if len(interactions) == 0:
        return elos

    if len(players) != len(elos):
        raise ValueError(f"Players and elos length mismatch\
: {len(players)} != {len(elos)}")

    for elo in elos:
        if not isinstance(elo, EloRate):
            raise TypeError("elos must be of type list[EloRate]")

    players_in_interactions = set()

    for interaction in interactions:
        players_in_interactions = \
            players_in_interactions.union(interaction.players)
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Bayeselo only accepts interactions involving \
both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player \
list")

        if interaction.outcomes[0] not in (0, .5, 1) or \
           interaction.outcomes[1] not in (0, .5, 1) or \
           sum(interaction.outcomes) != 1:
            raise Warning("Bayeselo takes outcomes in the (1, 0), (0, 1), \
(.5, .5) format, other values may have unspecified behavior")

    for e in elos:
        if e.base != elo_base or e.spread != elo_spread:
            raise ValueError(f"Elos with different bases and \
spreads are not compatible (expected base {elo_base}, spread {elo_spread} but \
got base {e.base}, spread {e.spread})")

    players_in_interactions = [
        player for player in players
        if player in players_in_interactions
    ]
    elos_to_update = [
        elo for elo, player in zip(elos, players)
        if player in players_in_interactions
    ]

    pairwise_stats = PopulationPairwiseStatistics.from_interactions(
        players=players_in_interactions,
        interactions=interactions
    )

    bradley_terry = BayesEloRating(
        pairwise_stats, elos=elos_to_update, elo_draw=elo_draw,
        elo_advantage=elo_advantage,
        base=elo_base, spread=elo_spread
    )

    bradley_terry.minorize_maximize(
        learn_home_field_bias=False,
        home_field_bias=elo_base ** (elo_advantage / elo_spread),
        learn_draw_bias=False,
        draw_bias=elo_base ** (elo_draw / elo_spread),
        iterations=iterations,
        tolerance=tolerance
    )

    bradley_terry.rescale_elos()

    new_elos = []
    for i, p in enumerate(players):
        if p in players_in_interactions:
            new_elos.append(bradley_terry.elos[0])
            bradley_terry.elos = bradley_terry.elos[1:]
        else:
            new_elos.append(elos[i])

    return new_elos
