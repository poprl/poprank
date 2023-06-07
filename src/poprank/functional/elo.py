from popcore import Interaction
from poprank import Rate


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[Rate]", k_factor: float,
) -> "list[Rate]":

    expected_scores = [.0 for player in players]
    true_scores = [.0 for player in players]

    for interaction in interactions:
        p1_elo = elos[players.index(interaction.players[0])].mu
        p2_elo = elos[players.index(interaction.players[1])].mu

        expected_scores[players.index(interaction.players[0])] += \
            1/(1+10**((p2_elo-p1_elo)/400))
        expected_scores[players.index(interaction.players[1])] += \
            1/(1+10**((p1_elo-p2_elo)/400))
        
        true_scores[players.index(interaction.players[0])] += \
            interaction.outcomes[0]
        true_scores[players.index(interaction.players[1])] += \
            interaction.outcomes[1]

    Rates = [Rate(e.mu + k_factor*(true_scores[i] - expected_scores[i]), 0)
             for i, e in enumerate(elos)]

    return Rates


def bayeselo(
    players: "list[str]", interactions: "list[Interaction]", elos: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
