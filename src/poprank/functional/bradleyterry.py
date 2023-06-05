from popcore import Interaction
from poprank import Rate


def bradleyterry(
    players: "list[str]", interactions: "list[Interaction]", 
    ratings: "list[Rate]", iterations: int, normalized: bool
) -> "list[Rate]":
    """_summary_ ref: https://doi.org/10.1214/aos/1079120141

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    raise NotImplementedError()


def bradleyterry_with_context(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[float]", iterations: int, theta_0: float, normalize: bool,
):
    """_summary_

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (list[float]): _description_
        iterations (int): _description_
        theta_0 (float): _description_
        normalize (bool): _description_
    """
    raise NotImplementedError()


def bradleyterry_with_context_draw(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[float]", iterations: int, theta_0: float, normalize: bool,
):
    raise NotImplementedError()
