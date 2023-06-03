from popcore import Interaction
from poprank import Rate


def windrawlose(
    players: list[str], interactions: list[Interaction], ratings: list[float],
    win_value: float, draw_value: float, loss_value: float
) -> list[Rate]:
    """_summary_

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (list[float]): _description_
        win_value (float): _description_
        draw_value (float): _description_
        loss_value (float): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        list[float]: _description_
    """
    raise NotImplementedError()


def winlose(
    players: list[str], interactions: list[Interaction], ratings: list[float],
    win_value: float, draw_value: float, loss_value: float
) -> list[Rate]:
    """_summary_

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (list[float]): _description_
        win_value (float): _description_
        draw_value (float): _description_
        loss_value (float): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        list[float]: _description_
    """
    raise NotImplementedError()
