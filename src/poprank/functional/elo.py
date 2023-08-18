from math import log
from dataclasses import dataclass
from popcore import Interaction
from poprank import Rate, EloRate
from poprank.functional.wdl import windrawlose


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float = 20, wdl: bool = False
) -> "list[EloRate]":
    """Rates players by calculating their new elo after a set of interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5).

    See also: :meth:`poprank.functional.elo.bayeselo`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
            get a rating from. Every interaction should be between exactly 2
            players and result in a win (1, 0), a loss (0, 1)
            or a draw (0.5, 0.5)
        elos (list[EloRate]): the initial ratings of the players
        k_factor (float): maximum possible adjustment per game. Larger means
            player rankings change faster
        wdl (bool, optional): Turn the interactions into the (1, 0), (.5, .5),
            (0, 1) format automatically. Defaults to False.
    Raises:
        ValueError: if the numbers of players and ratings don't match,
            if an interaction has the wrong number of players,
            if an interaction has the wrong number of outcomes,
            if a player that does not appear in `players`is in an
            interaction
        TypeError: Using Rate instead of EloRate

    Returns:
        list[EloRate]: the updated ratings of all players
    """

    # Checks
    if len(players) != len(elos):
        raise ValueError(f"Players and elos length mismatch\
: {len(players)} != {len(elos)}")

    for elo in elos:
        if not isinstance(elo, EloRate):
            raise TypeError("elos must be of type list[EloRate]")

    for interaction in interactions:
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Elo only accepts interactions involving \
both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player \
list")

        if not wdl and (interaction.outcomes[0] not in (0, .5, 1) or
                        interaction.outcomes[1] not in (0, .5, 1) or
                        sum(interaction.outcomes) != 1):
            raise Warning("Elo takes outcomes in the (1, 0), (0, 1), (.5, .5) \
format, other values may have unspecified behavior \
(set wdl=True to automatically turn interactions \
into the windrawlose format)")

    # Calculate the expected score vs true score of all players in the given
    # set of interactions and adjust elo afterwards accordingly.
    expected_scores: "list[float]" = [.0 for player in players]
    true_scores: "list[float]" = [.0 for player in players]

    for interaction in interactions:
        id_p1 = players.index(interaction.players[0])
        id_p2 = players.index(interaction.players[1])

        expected_scores[id_p1] += elos[id_p1].expected_outcome(elos[id_p2])
        expected_scores[id_p2] += elos[id_p2].expected_outcome(elos[id_p1])

        true_scores[players.index(interaction.players[0])] += \
            interaction.outcomes[0]
        true_scores[players.index(interaction.players[1])] += \
            interaction.outcomes[1]

    if wdl:
        true_scores = [r.mu for r in
                       windrawlose(players=players,
                                   interactions=interactions,
                                   ratings=[Rate(0, 0) for p in players],
                                   win_value=1,
                                   draw_value=.5,
                                   loss_value=0)]
    # New elo values
    rates: "list[EloRate]" = \
        [EloRate(e.mu + k_factor*(true_scores[i] - expected_scores[i]), e.std)
         for i, e in enumerate(elos)]

    return rates
