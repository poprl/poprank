from popcore import Interaction
from poprank import Rate


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[Rate]", k_factor: float,
) -> "list[Rate]":

    if len(players) != len(elos):
        raise ValueError(f"Players and elos length mismatch\
                           : {len(players)} != {len(elos)}")

    for interaction in interactions:
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Elo only accepts interactions involving \
                              both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player \
                              list")

        if interaction.outcomes[0] not in (0, .5, 1) or \
           interaction.outcomes[1] not in (0, .5, 1) or \
           sum(interaction.outcomes) != 1:
            raise Warning("Elo takes outcomes in the (1, 0), (0, 1), (.5, .5) \
                           format, other values may have unspecified behavior")

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
