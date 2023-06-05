from typing import Tuple
from popcore import Interaction
from poprank import Rate


def trueskill(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Tuple[float, float]]"
) -> "list[Rate]":
    raise NotImplementedError()


def trueskill2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Tuple[float, float]]"
) -> "list[Rate]":
    raise NotImplementedError()
