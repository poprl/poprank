from typing import List
from poprank.core import Game


def win_loss_rating(
        game: Game, win_value: int, loss_value: int) -> List[int]:
    """_summary_

    Args:
        game (Game): _description_
        win_value (int): _description_
        loss_value (int): _description_

    Returns:
        List[int]: _description_
    """
    ratings = [player.rating for player in game.players]
    assert len(ratings) == 2, "only defined for two-player games"

    p1_score, p2_score = game.scores

    if p1_score > p2_score:
        ratings[0] += win_value
        ratings[1] += loss_value
    elif p2_score > p1_score:
        ratings[0] += loss_value
        ratings[1] += win_value
    else:
        raise NotImplementedError()

    return ratings


def win_draw_loss_rating(
        game: Game, win_value: int,
        draw_value: int, loss_value: int) -> list[int]:
    """_summary_

    Args:
        game (Game): _description_
        win_value (int): _description_
        draw_value (int): _description_
        loss_value (int): _description_

    Returns:
        list[int]: _description_
    """
    ratings = [player.rating for player in game.players]
    assert len(ratings) == 2, "only define for two-player games"

    p1_score, p2_score = game.scores

    if p1_score > p2_score:
        ratings[0] += win_value
        ratings[1] += loss_value
    elif p2_score > p1_score:
        ratings[0] += loss_value
        ratings[1] += win_value
    else:
        ratings[0] += draw_value
        ratings[1] += draw_value

    return ratings
