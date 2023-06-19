from math import sqrt
from popcore import Interaction, Team, Player
from poprank import Rate
from copy import deepcopy
from scipy.stats import norm
from typing import Callable


def trueskill(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]", tau: float = 25./300., beta: float = 25./6.,
    draw_probability: float = .1, tolerance: float = 0.0001,
    weights: "list[list[float]]" = None
) -> "list[Rate]":

    # Code here adapted from copyrighted material by Heungsub Lee

    """Full license it's under:
    All the Python code and the documentation in this TrueSkill project is
    Copyright (c) 2012-2016 by Heungsub Lee. All rights reserved.

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    * The names of the contributors may not be used to endorse or
    promote products derived from this software without specific
    prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    # TODO: Add checks

    new_ratings: "list[list[Rate]]" = \
        [[deepcopy(rating)] if not isinstance(rating, list) else
         [deepcopy(r) for r in rating] for rating in ratings]
    teams: list[Team] = \
        [p if isinstance(p, Team) else
         (Team(name=p.name, members=[p]) if isinstance(p, Player) else
         (Team(name=p, members=[p]))) for p in players]
    team_names: list[str] = [t.name for t in teams]

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

        sorted_ratings: list[Rate] = [new_ratings[team_names.index(p)]
                                      for p in sorted_players]
        sorted_teams: list[Team] = [teams[team_names.index(p)].members
                                    for p in sorted_players]

        # ------ Factor graph objects ------ #

        flat_ratings: list[Rate] = [item for sub_list in sorted_ratings
                                    for item in sub_list]

        if weights is None:
            flat_weights = [1. for x in flat_ratings]
        else:
            sorted_weights: list[list[float]]
            sorted_weights = [weights[team_names.index(p)]
                              for p in sorted_players]
            flat_weights = [item for sub_list in sorted_weights
                            for item in sub_list]

        INF: float = float("inf")

        class Gaussian(object):
            """A model for the normal distribution."""

            # Precision, the inverse of the variance.
            pi: float = 0.
            # Precision adjusted mean, the precision multiplied by the mean.
            tau: float = 0.

            def __init__(self, mu: float = None, sigma: float = None,
                         pi: float = 0, tau: float = 0) -> None:
                if mu is not None:  # Note: sigma should be nonzero
                    pi = sigma ** -2
                    tau = pi * mu
                self.pi = pi
                self.tau = tau

            @property
            def mu(self) -> float:
                """A property which returns the mean."""
                return self.pi and self.tau / self.pi

            @property
            def sigma(self) -> float:
                """A property which returns the the std deviation"""
                return sqrt(1 / self.pi) if self.pi else INF

            def __mul__(self, other: "Gaussian") -> "Gaussian":
                """Multiplication between two Gaussians"""
                pi, tau = self.pi + other.pi, self.tau + other.tau
                return Gaussian(pi=pi, tau=tau)

            def __truediv__(self, other: "Gaussian") -> "Gaussian":
                """Division between two Gaussians"""
                pi, tau = self.pi - other.pi, self.tau - other.tau
                return Gaussian(pi=pi, tau=tau)

        class Variable(Gaussian):
            """A variable in the factor graph"""

            def __init__(self) -> None:
                self.messages: dict["Factor", Gaussian] = {}
                super(Variable, self).__init__()

            def set(self, value: Gaussian) -> float:
                delta: float = self.delta(value)
                self.pi, self.tau = value.pi, value.tau
                return delta

            def delta(self, other: Gaussian) -> float:
                pi_delta: float = abs(self.pi - other.pi)
                if pi_delta == INF:
                    return 0.
                return max(abs(self.tau - other.tau), sqrt(pi_delta))

            def update_message(self, factor: "Factor", pi: float = 0.,
                               tau: float = 0) -> float:
                message: Gaussian = Gaussian(pi=pi, tau=tau)
                old_message: Gaussian = self.messages[factor]
                self.messages[factor] = message
                return self.set(self / old_message * message)

            def update_value(self, factor: "Factor", pi: float = 0,
                             tau: float = 0, value: Gaussian = None) -> float:
                value = value or Gaussian(pi=pi, tau=tau)
                old_message: Gaussian = self.messages[factor]
                self.messages[factor] = value * old_message / self
                return self.set(value)

        class Factor():

            def __init__(self, variables: list[Variable]) -> None:
                self.variables: list[Variable] = variables
                for variable in variables:
                    variable.messages[self] = Gaussian()

            def pass_message_down(self) -> float:
                return 0.

            def pass_message_up(self) -> float:
                return 0.

            @property
            def variable(self) -> Variable:
                return self.variables[0]

        class PriorFactor(Factor):

            def __init__(self, variable: Variable, rating: Rate,
                         dynamic_variance: float = 0.) -> None:
                super(PriorFactor, self).__init__([variable])
                self.rating: Rate = rating
                self.dynamic_variance: float = dynamic_variance

            def pass_message_down(self) -> float:
                sigma: float = sqrt(self.rating.std ** 2 +
                                    self.dynamic_variance ** 2)
                value: Gaussian = Gaussian(self.rating.mu, sigma)
                return self.variable.update_value(self, value=value)

        class LikelihoodFactor(Factor):

            def __init__(self, mean_variable: Variable,
                         value_variable: Variable, variance: float) -> None:
                super(LikelihoodFactor, self).__init__([mean_variable,
                                                        value_variable])
                self.mean: Variable = mean_variable
                self.value: Variable = value_variable
                self.variance: Variable = variance

            def pass_message_down(self) -> float:
                """Update value."""
                msg: Gaussian = self.mean / self.mean.messages[self]
                a: float = 1. / (1. + self.variance * msg.pi)
                return self.value.update_message(self, a * msg.pi, a * msg.tau)

            def pass_message_up(self) -> float:
                """Update mean."""
                msg: Gaussian = self.value / self.value.messages[self]
                a: float = 1. / (1. + self.variance * msg.pi)
                return self.mean.update_message(self, a * msg.pi, a * msg.tau)

        class SumFactor(Factor):

            def __init__(self, sum_variable: Variable,
                         term_variables: list[Variable],
                         weights: list[int]):
                super(SumFactor, self).__init__([sum_variable] +
                                                term_variables)
                self.sum: Variable = sum_variable
                self.terms: list[Variable] = term_variables
                self.weights: list[int] = weights

            def pass_message_down(self) -> float:
                msgs: list[Gaussian] = [var.messages[self] for var
                                        in self.terms]
                return self.update(self.sum, self.terms, msgs, self.weights)

            def pass_message_up(self, index: int = 0) -> float:
                weight: float = self.weights[index]
                weights: list[float] = []
                for i, w in enumerate(self.weights):
                    weights.append(0. if weight == 0
                                   else 1. / weight if i == index
                                   else -w / weight)
                values = self.terms[:]
                values[index] = self.sum
                msgs = [var.messages[self] for var in values]
                return self.update(self.terms[index], values, msgs, weights)

            def update(self, variable: Variable, values: list[Variable],
                       msgs: list[Gaussian], weights: list[float]) -> float:
                pi_inv: float = 0
                mu: float = 0
                for value, msg, weight in zip(values, msgs, weights):
                    div: float = value / msg
                    mu += weight * div.mu
                    if pi_inv == INF:
                        continue
                    pi_inv += INF if float(div.pi) == 0 else \
                        weight ** 2 / float(div.pi)
                pi: float = 1. / pi_inv
                tau: float = pi * mu
                return variable.update_message(self, pi, tau)

        class TruncateFactor(Factor):

            def __init__(self, variable: Variable,
                         v_func: Callable[[float, float], float],
                         w_func: Callable[[float, float], float],
                         draw_margin: float):
                super(TruncateFactor, self).__init__([variable])
                self.v_func: Callable[[float, float], float] = v_func
                self.w_func: Callable[[float, float], float] = w_func
                self.draw_margin: float = draw_margin

            def pass_message_up(self) -> float:
                div: Gaussian = self.variable / self.variable.messages[self]
                sqrt_pi: float = sqrt(div.pi)
                diff: float = div.tau / sqrt_pi
                draw_margin: float = self.draw_margin * sqrt_pi
                v: float = self.v_func(diff, draw_margin)
                w: float = self.w_func(diff, draw_margin)
                denom: float = 1. - w
                pi: float = div.pi / denom
                tau: float = (div.tau + sqrt_pi * v) / denom
                return self.variable.update_value(self, pi, tau)

        # Create variables (gaussians)
        rating_variables: list[Variable] = [Variable() for x in flat_ratings]
        perf_variables: list[Variable] = [Variable() for x in flat_ratings]
        team_perf_variables: list[Variable] = \
            [Variable() for x in sorted_ratings]
        team_diff_variables: list[Variable] = \
            [Variable() for x in sorted_ratings[:-1]]
        team_sizes: list[int] = [len(t) for t in sorted_teams]
        team_sizes = [sum(team_sizes[0:i+1]) for i, _ in enumerate(team_sizes)]

        def v_win(diff: float, draw_margin: float) -> float:
            """The non-draw version of "V" function.  "V" calculates a
            variation of a mean.
            """
            x: float = diff - draw_margin
            denom: float = norm.cdf(x)
            return (norm.pdf(x) / denom) if denom else -x

        def v_draw(diff: float, draw_margin: float) -> float:
            """The draw version of "V" function."""
            abs_diff: float = abs(diff)
            a: float = draw_margin - abs_diff
            b: float = -draw_margin - abs_diff
            denom: float = norm.cdf(a) - norm.cdf(b)
            numer: float = norm.pdf(b) - norm.pdf(a)
            return ((numer / denom) if denom else a) * (-1 if diff < 0 else +1)

        def w_win(diff: float, draw_margin: float) -> float:
            """The non-draw version of "W" function.  "W" calculates a
            variation of a standard deviation.
            """
            x: float = diff - draw_margin
            v: float = v_win(diff, draw_margin)
            w: float = v * (v + x)
            if 0 < w < 1:
                return w
            raise FloatingPointError()

        def w_draw(diff: float, draw_margin: float) -> float:
            """The draw version of "W" function."""
            abs_diff: float = abs(diff)
            a: float = draw_margin - abs_diff
            b: float = -draw_margin - abs_diff
            denom: float = norm.cdf(a) - norm.cdf(b)
            if denom == 0.:
                raise FloatingPointError()
            v: float = v_draw(abs_diff, draw_margin)
            return (v ** 2) + (a * norm.pdf(a) - b * norm.pdf(b)) / denom

        # ------ Build factor graph ------ #

        rating_layer: list[PriorFactor] = \
            [PriorFactor(rating_var, rating, tau) for rating_var,
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
            draw_margin: float = norm.ppf((draw_probability + 1) / 2.) \
                * sqrt(size) * beta
            v_func: Callable[[float, float], float]
            w_func: Callable[[float, float], float]
            v_func, w_func = (v_draw, w_draw) if ranks[i] == ranks[i + 1] \
                else (v_win, w_win)  # Handle draws
            trunc_layer.append(TruncateFactor(team_diff_var, v_func,
                                              w_func, draw_margin))

        """layers = [rating_layer, perf_layer, team_perf_layer,
                  team_diff_layer, trunc_layer]"""

        # ------ Iterative algorithm ------ #

        for factor in rating_layer + perf_layer + team_perf_layer:
            factor.pass_message_down()

        team_diff_len: int = len(team_diff_layer)

        for iteration in range(10):
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

        player_index: int = 0
        for team, team_ratings in enumerate(sorted_ratings):
            for individual, _ in enumerate(team_ratings):
                sorted_ratings[team][individual] = \
                    Rate(rating_layer[player_index].variable.mu,
                         rating_layer[player_index].variable.sigma)
                player_index += 1

    return new_ratings


def trueskill2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
