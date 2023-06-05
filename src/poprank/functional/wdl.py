from popcore import Interaction
from poprank import Rate


def typecheck(item, expected_type):
    if not issubclass(type(item), expected_type):
        raise TypeError(f"Expected {expected_type} but got\
            {type(item)}")


def windrawlose(
    players: "list[str]",
    interactions: "list[Interaction]", ratings: "list[float]",
    win_value: float, draw_value: float, loss_value: float
) -> "list[Rate]":
    """_summary_

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (list[float]): _description_
        win_value (float): _description_
        draw_value (float): _description_
        loss_value (float): _description_

    Raises:
        TypeError: _description_

    Returns:
        list[float]: _description_
    """

    # Typechecking
    for player in players:
        typecheck(player, str)

    for interaction in interactions:
        typecheck(interaction, Interaction)

    for rating in ratings:
        typecheck(rating, (float, int))

    typecheck(win_value, (float, int))
    typecheck(draw_value, (float, int))
    typecheck(loss_value, (float, int))

    ranks = [Rate(rating, 0) for rating in ratings]

    for interaction in interactions:
        best_score: int = max(interaction.outcomes)
        indices: "list[int]" = [i for
                                i, score in enumerate(interaction.outcomes)
                                if score == best_score]

        if len(indices) > 1:
            for i in indices:
                player_index = players.index(interaction.players[i])
                new_mu = ranks[player_index].mu + draw_value
                ranks[player_index] = Rate(new_mu, 0)

        else:
            player_index = players.index(interaction.players[indices[0]])
            new_mu = ranks[player_index].mu + win_value
            ranks[player_index] = Rate(new_mu, 0)

        for i, player in enumerate(interaction.players):
            if interaction.outcomes[i] < best_score:
                player_index = players.index(player)
                new_mu = ranks[player_index].mu + loss_value
                ranks[player_index] = Rate(new_mu, 0)

    return ranks


def winlose(
    players: "list[str]",
    interactions: "list[Interaction]", ratings: "list[float]",
    win_value: float, loss_value: float
) -> "list[Rate]":
    """_summary_

    Args:
        players (list[str]): _description_
        interactions (list[Interaction]): _description_
        ratings (list[float]): _description_
        win_value (float): _description_
        loss_value (float): _description_

    Raises:
        TypeError: _description_

    Returns:
        list[float]: _description_
    """
    
    return windrawlose(players, interactions, ratings, win_value, win_value,
                       loss_value)
