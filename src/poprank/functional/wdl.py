from popcore import Interaction
from poprank import Rate

"""Rates players by awarding fixed points for wins, draws and losses."""


def _typecheck(item: "any", expected_type: type) -> None:
    """Typechecking for function arguments"""
    assert issubclass(type(item), expected_type), \
        f"Wrong argument type: Expected {expected_type} but got {type(item)}"


def windrawlose(
    players: "list[str]",
    interactions: "list[Interaction]", ratings: "list[Rate]",
    win_value: float, draw_value: float, loss_value: float
) -> "list[Rate]":
    """
    Rates players by awarding fixed points for wins, draws and losses.

    Works for N players interactions, where

    -If a single player has the max amount of points in a given interaction,
    they get `win_value` added to their rating

    -If multiple players have the max amount of points in a given interaction,
    they get `draw_value` added to their rating

    -If a player does not have the max amount of points in a given interaction,
    they get `loss_value` added to their rating

    :param list[str] players: A list containing all unique player identifiers
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from
    :param list[Rate] ratings: The initial ratings of the players
    :param float win_value: The points awarded for a win
    :param float draw_value: The points awarded for a draw
    :param float loss_value: The points awarded for a loss

    :raises AssertionError: If the arguments are of the wrong type
    :raises ValueError: If the numbers of players and ratings don't match

    :return: The updated ratings of all players
    :rtype: list[Rate]

    Example
    -------

    .. code-block:: python

        # Test windrawlose in a N agent setting

        from poprank.functional.wdl import windrawlose
        from poprank import Rate
        from popcore import Interaction

        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[5, 5, 4, 4, 1]
            ),
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[0, 0, 2, 0, 1]
            )
        ]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        new_ratings = windrawlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings],
                win_value=3, draw_value=1, loss_value=0
            )

        # new_ratings will be
        # [Rate(1, 0), Rate(1, 0), Rate(3, 0), Rate(0, 0), Rate(0, 0)]

    .. seealso::
        :meth:`poprank.functional.winlose`
    """

    for player in players:
        _typecheck(player, str)

    for interaction in interactions:
        _typecheck(interaction, Interaction)

    for rating in ratings:
        _typecheck(rating, (Rate, int))

    _typecheck(win_value, (float, int))
    _typecheck(draw_value, (float, int))
    _typecheck(loss_value, (float, int))

    if len(players) != len(ratings):
        raise ValueError("Players and ratings length mismatch"
                         f": {len(players)} != {len(ratings)}")

    # Value to return: the rates of all agents
    rates: "list[Rate]" = [rating for rating in ratings]

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
    interactions: "list[Interaction]", ratings: "list[Rate]",
    win_value: float, loss_value: float
) -> "list[Rate]":
    """Rates players by awarding fixed points for wins and losses.

    Works for N players interactions, where

    -If a single player has the max amount of points in a given interaction,
    they get `win_value` added to their rating

    -If multiple players have the max amount of points in a given interaction,
    they get `win_value` added to their rating

    -If a player does not have the max amount of points in a given interaction,
    they get `loss_value added` to their rating

    :param list[str] players: A list containing all unique player identifiers
    :param list[Interaction] interactions: A list containing the interactions to
        get a rating from
    :param list[Rate] ratings: The initial ratings of the players
    :param float win_value: The points awarded for a win
    :param float loss_value: The points awarded for a loss

    :raises AssertionError: If the arguments are of the wrong type
    :raises ValueError: If the numbers of players and ratings don't match

    :return: the updated ratings of all players
    :rtype: list[Rate]

    Example
    -------

    .. code-block:: python

        # Test winlose in N agent setting

        from poprank.functional.wdl import winlose
        from poprank import Rate
        from popcore import Interaction

        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[5, 5, 4, 4, 1]
            ),
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[0, 0, 2, 0, 1]
            )
        ]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        new_ratings = winlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings],
                win_value=3, loss_value=0
            )
        
        # new_ratings will be
        # [Rate(3, 0), Rate(3, 0), Rate(3, 0), Rate(0, 0), Rate(0, 0)]

    .. seealso::
        :meth:`poprank.functional.windrawlose`
    """

    return windrawlose(players, interactions, ratings, win_value, win_value,
                       loss_value)
