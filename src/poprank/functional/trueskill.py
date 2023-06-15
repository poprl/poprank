import numpy as np
from popcore import Interaction
from poprank import Rate
from copy import deepcopy


def trueskill(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]", tau: float = 25./600., beta: float = 25./6.,
    draw_probability: float = .1
) -> "list[Rate]":

    new_ratings = [deepcopy(rating) for rating in ratings]

    for interaction in interactions:
        # ------ Sort rating groups by rank ------ #

        # IMPORTANT: Here, we assume the interactions outcomes are scores,
        # and we must turn them into ranks
        sorted_players, sorted_outcomes = \
            zip(*sorted(zip(interaction.players, interaction.outcomes),
                        key=lambda x: x[1], reverse=True))

        sorted_outcomes = list(dict.fromkeys(sorted_outcomes))
        ranks = [sorted_outcomes.index(outcome) + 1 for outcome in
                 interaction.outcomes]

        sorted_ratings = [ratings[players.index(p)] for p in sorted_players]

        # TODO: Figure out weights
        sorted_weights = [1. for x in sorted_outcomes]

        # ------ Build factor graph ------ #
        args = (sorted_players, sorted_outcomes, sorted_weights)

        flat_ratings = np.ndarray.flatten(sorted_ratings)
        flat_weights = np.ndarray.flatten(sorted_weights)
        number_of_individuals = len(flat_ratings)
        number_of_teams = len(sorted_ratings)

        # Create variables
        # Variables are gaussians so we use rate
        rating_vars = [Rate(0., 1.) for x in flat_ratings]
        perf_vars = [Rate(0., 1.) for x in flat_ratings]
        team_perf_vars = [Rate(0., 1.) for x in sorted_ratings]
        team_diff_vars = [Rate(0., 1.) for x in sorted_ratings]
        team_sizes = [len(t) for t in sorted_players]
        team_sizes = [sum(team_sizes[0:i+1]) for i, _ in enumerate(team_sizes)]

        def build_rating_layer():
            for rating_var, rating in zip(rating_vars, flat_ratings):
                yield (rating_var, rating)  # PriorFactor
        
        def build_perf_layer():
            for rating_var, perf_var in zip(rating_vars, perf_vars):
                yield (rating_var, perf_var)  # LikelihoodFactor
        
        def build_team_perf_layer():
            for team, team_perf_var in enumerate(team_perf_vars):
                if team > 0:
                    start = team_sizes[team-1]
                else:
                    start = 0
                end = team_sizes[team]

                child_perf_vars = perf_vars[start:end]
                coefficients = flat_weights[start:end]

                yield (team_perf_var, child_perf_vars, coefficients)
        
        def build_team_diff_layer():
            for team, team_diff_var in enumerate(team_diff_vars):
                yield (team_diff_var, team_perf_vars[team:team+2], [1, -1])
        
        def build_trunc_layer():
            for x, team_diff_var in enumerate(team_diff_vars):
                #
                # TODO: Make if statement for dynamic draw probability
                #
                size = sum([len(x) for x in sorted_ratings[x:x+2]])
                

        # ------ Make result ------ #

        # Restore original ordering
    return new_ratings


def trueskill2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
