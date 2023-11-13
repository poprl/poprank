from copy import deepcopy
from math import sqrt
from typing import Callable
from statistics import NormalDist
from popcore import Interaction, Team, Player
from poprank import Rate
from ._trueskill.factor_graph import (
    Variable, PriorFactor, LikelihoodFactor, flatten,
    SumFactor, TruncateFactor, v_draw, v_win, w_draw, w_win
)


def trueskill(
    players: "list[Team]", interactions: "list[Interaction]",
    ratings: "list[list[Rate]]", dynamic_factor: float = 1./12.,
    beta: float = 25./6., draw_probability: float = .1,
    weights: "list[list[float]]" = None,
    iterations: int = 10, tolerance: float = 1e-4
) -> "list[Rate]":
    """Rates players by calculating their trueskill

    Given a set of interactions and initial trueskill ratings, uses a
    factor graph to create a new set of ratings.
    The iterative algorithm is performed for the number of
    specified iterations or until the changes are below the tolerance
    value, whichever comes first.
    Interactions outcomes are assumed to be scores, and are turned into
    rankings automatically.

    Args:
        players (list[Team]): The list of all teams of players
        interactions (list[Interactions]): The list of all interactions
        ratings (list[list[Rate]]): The initial ratings of the players
        dynamic_factor (float, optional): Default dynamic factor. Tau in the
            original paper. Defaults to 1.0/12.0.
        beta (float, optional): Default difference between two ratings that
            implies 76% chance of winning. Defaults to 25.0/6.0.
        draw_probability (float, optional): Probability of drawing.
            Defaults to 0.1.
        weights (list[list[float]], optional): Weight associated with each
            player. Defaults to None.
        iterations (int, optional): The maximum number of iterations the
            iterative algorithm will go through. Defaults to 10000.
        tolerance (float, optional): The error threshold below which the
            iterative algorithm stops. Defaults to 1e-4.

    Returns:
        list[list[Rate]]: The updated ratings of all players
    """

    # TODO: Add checks

    # The format of the input is messy, but it allows for great user
    # flexibility

    # We turn the rates, which can be list[list[Rate] | Rate] into
    # list[list[Rate]]. We must turn it back when returning

    new_ratings_reformatted: "list[list[Rate] | Rate]" = deepcopy(ratings)
    new_ratings_reformatted = [x if isinstance(x, list) else [x]
                               for x in new_ratings_reformatted]

    # We turn the players, which can be list[str | Player | Team] into
    # list[Team]
    teams: list[Team] = \
        [p if isinstance(p, Team) else
         (Team(name=p.name, members=[p]) if isinstance(p, Player) else
         (Team(name=p, members=[p]))) for p in players]
    team_names: list[str] = [t.name for t in teams]

    # We flatten the ratings for simplicity
    new_ratings: "list[Rate]" = flatten(new_ratings_reformatted)
    player_names: list[str] = [p for t in teams for p in t.members]

    for interaction in interactions:
        # ------ Sort rating groups by rank ------ #

        # IMPORTANT: Here, we assume the interactions outcomes are scores,
        # and we must turn them into ranks
        sorted_players: list[Player]
        sorted_outcomes: list[list[float]]
        sorted_players, sorted_outcomes = \
            zip(*sorted(zip(interaction.players, interaction.outcomes),
                        key=lambda x: x[1], reverse=True))

        sorted_outcomes = list(dict.fromkeys(sorted_outcomes))
        ranks: list[int] = [sorted_outcomes.index(outcome) + 1 for outcome in
                            interaction.outcomes]
        ranks.sort()

        sorted_ratings: list[Rate] = []
        sorted_teams: list[Team] = []
        for t in sorted_players:
            if isinstance(t, Team):
                sorted_ratings.append([new_ratings[player_names.index(p)]
                                       for p in t.members])
                sorted_teams.append([teams[player_names.index(p)].members[0]
                                     for p in t.members])
            else:
                sorted_ratings.append(
                    new_ratings_reformatted[team_names.index(t)])
                sorted_teams.append(teams[team_names.index(t)].members)

        # ------ Factor graph objects ------ #

        flat_ratings: list[Rate] = flatten(sorted_ratings)

        if weights is None:
            flat_weights = [1. for x in flat_ratings]
        else:
            sorted_weights: list[list[float]]
            sorted_weights = [weights[team_names.index(p)]
                              for p in sorted_players]
            flat_weights = flatten(sorted_weights)

        # Create variables (gaussians)
        rating_variables: list[Variable] = [Variable() for x in flat_ratings]
        perf_variables: list[Variable] = [Variable() for x in flat_ratings]
        team_perf_variables: list[Variable] = \
            [Variable() for x in sorted_ratings]
        team_diff_variables: list[Variable] = \
            [Variable() for x in sorted_ratings[:-1]]
        team_sizes: list[int] = [len(t) for t in sorted_teams]
        team_sizes = [sum(team_sizes[0:i+1]) for i, _ in enumerate(team_sizes)]

        # ------ Build factor graph ------ #

        rating_layer: list[PriorFactor] = \
            [PriorFactor(rating_var, rating, dynamic_factor) for rating_var,
             rating in zip(rating_variables, flat_ratings)]

        perf_layer: list[LikelihoodFactor] = \
            [LikelihoodFactor(rating_var, perf_var, beta ** 2) for
             rating_var, perf_var in zip(rating_variables, perf_variables)]

        team_perf_layer: list[SumFactor] = []
        for team, team_perf_var in enumerate(team_perf_variables):
            start: int = team_sizes[team-1] if team > 0 else 0
            end: int = team_sizes[team]

            child_perf_vars: list[Variable] = perf_variables[start:end]
            coefficients: list[Variable] = flat_weights[start:end]

            team_perf_layer.append(SumFactor(team_perf_var, child_perf_vars,
                                             coefficients))

        team_diff_layer: list[SumFactor] = \
            [SumFactor(team_diff_var, team_perf_variables[team:team+2],
                       [1, -1])
             for team, team_diff_var in enumerate(team_diff_variables)]

        trunc_layer: list[TruncateFactor] = []
        for i, team_diff_var in enumerate(team_diff_variables):
            # TODO: Make if statement for dynamic draw probability
            size: int = sum([len(x) for x in sorted_ratings[i:i+2]])
            draw_margin: float = \
                NormalDist().inv_cdf((draw_probability + 1) / 2.) \
                * sqrt(size) * beta
            v_func: Callable[[float, float], float]
            w_func: Callable[[float, float], float]
            v_func, w_func = (v_draw, w_draw) if ranks[i] == ranks[i + 1] \
                else (v_win, w_win)  # Handle draws
            trunc_layer.append(TruncateFactor(team_diff_var, v_func,
                                              w_func, draw_margin))

        # ------ Iterative algorithm ------ #

        for factor in rating_layer + perf_layer + team_perf_layer:
            factor.pass_message_down()

        team_diff_len: int = len(team_diff_layer)

        for iteration in range(iterations):
            if team_diff_len == 1:
                # only two teams
                team_diff_layer[0].pass_message_down()
                delta = trunc_layer[0].pass_message_up()
            else:
                # multiple teams
                delta: float = 0.
                for iteration in range(team_diff_len - 1):
                    team_diff_layer[iteration].pass_message_down()
                    delta = max(delta,
                                trunc_layer[iteration].pass_message_up())

                    # up to right variable
                    team_diff_layer[iteration].pass_message_up(1)
                for iteration in range(team_diff_len - 1, 0, -1):
                    team_diff_layer[iteration].pass_message_down()
                    delta = max(delta,
                                trunc_layer[iteration].pass_message_up())

                    # up to left variable
                    team_diff_layer[iteration].pass_message_up(0)

            if delta <= tolerance:
                break
        # up both ends
        team_diff_layer[0].pass_message_up(0)
        team_diff_layer[team_diff_len - 1].pass_message_up(1)
        # up the remainder of the black arrows
        for factor in team_perf_layer:
            for iteration in range(len(factor.variables) - 1):
                factor.pass_message_up(iteration)
        for factor in perf_layer:
            factor.pass_message_up()

        # ------ Update the new ratings ------ #

        flat_indices = [player_names.index(p) for t in sorted_teams for p in t]
        # Update the flat array contining all ratings
        for i, j in enumerate(flat_indices):
            new_ratings[j] = Rate(rating_layer[i].variable.mu,
                                  rating_layer[i].variable.std)

        # Update new_ratings_reformatted
        player_index: int = 0
        for team in new_ratings_reformatted:
            for i, p in enumerate(team):
                if player_index in flat_indices:
                    tmp = flat_indices.index(player_index)
                    team[i] = Rate(rating_layer[tmp].variable.mu,
                                   rating_layer[tmp].variable.std)
                player_index += 1

    # return the ratings to their original format
    for i, x in enumerate(ratings):
        if not isinstance(x, list):
            new_ratings_reformatted[i] = new_ratings_reformatted[i][0]

    return new_ratings_reformatted


def trueskill2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
