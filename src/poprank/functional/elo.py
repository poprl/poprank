from popcore import Interaction
from poprank import Rate


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[Rate]", k_factor: float,
) -> "list[Rate]":
    raise NotImplementedError()


def bayeselo(
    players: "list[str]", interactions: "list[Interaction]", elos: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
