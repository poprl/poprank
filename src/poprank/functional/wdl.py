from popcore import Interaction
from poprank import Rate


def typecheck(item: "any", expected_type: type) -> None:
    """Typechecking for function arguments"""
    assert issubclass(type(item), expected_type), \
        f"Wrong argument type: Expected {expected_type} but got {type(item)}"


def windrawlose(
    players: "list[str]",
    interactions: "list[Interaction]", ratings: "list[float]",
    win_value: float, draw_value: float, loss_value: float
) -> "list[Rate]":
    """Rates players by awarding fixed points for wins, draws and losses.

    Works for N players interactions, where
    -If a single player has the max amount of points in a given interaction,
    they get win_value added to their rating
    -If multiple players have the max amount of points in a given interaction,
    they get draw_value added to their rating
    -If a player does not have the max amount of points in a given interaction,
    they get loss_value added to their rating

    See also: :meth:`poprank.functional.winlose`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
                                          get a rating from
        ratings (list[float]): the initial ratings of the players
        win_value (float): the points awarded for a win
        draw_value (float): the points awarded for a draw
        loss_value (float): the points awarded for a loss

    Raises:
        TypeError: if the arguments are of the wrong type
        ValueError: if the numbers of players and ratings don't match

    Returns:
        list[Rate]: List of Rates
    """

    for player in players:
        typecheck(player, str)

    for interaction in interactions:
        typecheck(interaction, Interaction)

    for rating in ratings:
        typecheck(rating, (float, int))

    typecheck(win_value, (float, int))
    typecheck(draw_value, (float, int))
    typecheck(loss_value, (float, int))

    if len(players) != len(ratings):
        raise ValueError(f"Players and ratings length mismatch\
                           : {len(players)} != {len(ratings)}")

    # Value to return: the rates of all agents
    rates: "list[Rate]" = [Rate(rating, 0) for rating in ratings]

    # Update the rates for each interaction
    for interaction in interactions:
        best_score: int = max(interaction.outcomes)
        indices: "list[int]" = [i for
                                i, score in enumerate(interaction.outcomes)
                                if score == best_score]

        # If multiple players have the best score they all get the draw points
        if len(indices) > 1:
            for i in indices:
                player_index: int = players.index(interaction.players[i])
                new_mu: float = rates[player_index].mu + draw_value
                rates[player_index] = Rate(new_mu, 0)

        # If a single player has the best score they gets the win points
        else:
            player_index: int = players.index(interaction.players[indices[0]])
            new_mu: float = rates[player_index].mu + win_value
            rates[player_index] = Rate(new_mu, 0)

        # All the other players get the loss points
        for i, player in enumerate(interaction.players):
            if interaction.outcomes[i] < best_score:
                player_index: int = players.index(player)
                new_mu: float = rates[player_index].mu + loss_value
                rates[player_index] = Rate(new_mu, 0)

    return rates


def winlose(
    players: "list[str]",
    interactions: "list[Interaction]", ratings: "list[float]",
    win_value: float, loss_value: float
) -> "list[Rate]":
    """Rates players by awarding fixed points for wins and losses.

    Works for N players interactions, where
    -If a single player has the max amount of points in a given interaction,
    they get win_value added to their rating
    -If multiple players have the max amount of points in a given interaction,
    they get win_value added to their rating
    -If a player does not have the max amount of points in a given interaction,
    they get loss_value added to their rating

    See also: :meth:`poprank.functional.windrawlose`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
                                          get a rating from
        ratings (list[float]): the initial ratings of the players
        win_value (float): the points awarded for a win
        loss_value (float): the points awarded for a loss

    Raises:
        TypeError: if the arguments are of the wrong type
        ValueError: if the numbers of players and ratings don't match

    Returns:
        list[Rate]: List of Rates
    """

    return windrawlose(players, interactions, ratings, win_value, win_value,
                       loss_value)
