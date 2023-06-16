from math import sqrt
from popcore import Interaction, Team, Player
from poprank import Rate
from copy import deepcopy
from scipy.stats import norm


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

    new_ratings = [[deepcopy(rating)] if not isinstance(rating, list) else
                   [deepcopy(r) for r in rating] for rating in ratings]
    teams = [p if isinstance(p, Team) else
             (Team(name=p.name, members=[p]) if isinstance(p, Player) else
             (Team(name=p, members=[p]))) for p in players]
    team_names = [t.name for t in teams]

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

        sorted_ratings = [new_ratings[team_names.index(p)]
                          for p in sorted_players]
        sorted_teams = [teams[team_names.index(p)].members
                        for p in sorted_players]

        # ------ Factor graph objects ------ #

        flat_ratings = [item for sub_list in sorted_ratings
                        for item in sub_list]

        if weights is None:
            flat_weights = [1. for x in flat_ratings]
        else:
            sorted_weights = [weights[team_names.index(p)]
                              for p in sorted_players]
            flat_weights = [item for sub_list in sorted_weights
                            for item in sub_list]

        inf = float("inf")

        class Gaussian(object):
            """A model for the normal distribution."""

            #: Precision, the inverse of the variance.
            pi = 0
            #: Precision adjusted mean, the precision multiplied by the mean.
            tau = 0

            def __init__(self, mu=None, sigma=None, pi=0, tau=0):
                if mu is not None:
                    if sigma is None:
                        raise TypeError('sigma argument is needed')
                    elif sigma == 0:
                        raise ValueError('sigma**2 should be greater than 0')
                    pi = sigma ** -2
                    tau = pi * mu
                self.pi = pi
                self.tau = tau

            @property
            def mu(self):
                """A property which returns the mean."""
                return self.pi and self.tau / self.pi

            @property
            def sigma(self):
                """A property which returns the the square root of the
                variance."""
                return sqrt(1 / self.pi) if self.pi else inf

            def __mul__(self, other):
                pi, tau = self.pi + other.pi, self.tau + other.tau
                return Gaussian(pi=pi, tau=tau)

            def __truediv__(self, other):
                pi, tau = self.pi - other.pi, self.tau - other.tau
                return Gaussian(pi=pi, tau=tau)

        class Variable(Gaussian):

            def __init__(self):
                self.messages = {}
                super(Variable, self).__init__()

            def set(self, val):
                delta = self.delta(val)
                self.pi, self.tau = val.pi, val.tau
                return delta

            def delta(self, other):
                pi_delta = abs(self.pi - other.pi)
                if pi_delta == inf:
                    return 0.
                return max(abs(self.tau - other.tau), sqrt(pi_delta))

            def update_message(self, factor, pi=0, tau=0, message=None):
                message = message or Gaussian(pi=pi, tau=tau)
                old_message, self[factor] = self[factor], message
                return self.set(self / old_message * message)

            def update_value(self, factor, pi=0, tau=0, value=None):
                value = value or Gaussian(pi=pi, tau=tau)
                old_message = self[factor]
                self[factor] = value * old_message / self
                return self.set(value)

            def __getitem__(self, factor):
                return self.messages[factor]

            def __setitem__(self, factor, message):
                self.messages[factor] = message

        class Factor():

            def __init__(self, variables):
                self.vars = variables
                for var in variables:
                    var[self] = Gaussian()

            def down(self):
                return 0

            def up(self):
                return 0

            @property
            def var(self):
                assert len(self.vars) == 1
                return self.vars[0]

        class PriorFactor(Factor):

            def __init__(self, var, val, dynamic=0):
                super(PriorFactor, self).__init__([var])
                self.val = val
                self.dynamic = dynamic

            def down(self):
                sigma = sqrt(self.val.std ** 2 + self.dynamic ** 2)
                value = Gaussian(self.val.mu, sigma)
                return self.var.update_value(self, value=value)

        class LikelihoodFactor(Factor):

            def __init__(self, mean_var, value_var, variance):
                super(LikelihoodFactor, self).__init__([mean_var, value_var])
                self.mean = mean_var
                self.value = value_var
                self.variance = variance

            def calc_a(self, var):
                return 1. / (1. + self.variance * var.pi)

            def down(self):
                # update value.
                msg = self.mean / self.mean[self]
                a = self.calc_a(msg)
                return self.value.update_message(self, a * msg.pi, a * msg.tau)

            def up(self):
                # update mean.
                msg = self.value / self.value[self]
                a = self.calc_a(msg)
                return self.mean.update_message(self, a * msg.pi, a * msg.tau)

        class SumFactor(Factor):

            def __init__(self, sum_var, term_vars, coeffs):
                super(SumFactor, self).__init__([sum_var] + term_vars)
                self.sum = sum_var
                self.terms = term_vars
                self.coeffs = coeffs

            def down(self):
                vals = self.terms
                msgs = [var[self] for var in vals]
                return self.update(self.sum, vals, msgs, self.coeffs)

            def up(self, index=0):
                coeff = self.coeffs[index]
                coeffs = []
                for x, c in enumerate(self.coeffs):
                    try:
                        if x == index:
                            coeffs.append(1. / coeff)
                        else:
                            coeffs.append(-c / coeff)
                    except ZeroDivisionError:
                        coeffs.append(0.)
                vals = self.terms[:]
                vals[index] = self.sum
                msgs = [var[self] for var in vals]
                return self.update(self.terms[index], vals, msgs, coeffs)

            def update(self, var, vals, msgs, coeffs):
                pi_inv = 0
                mu = 0
                for val, msg, coeff in zip(vals, msgs, coeffs):
                    div = val / msg
                    mu += coeff * div.mu
                    if pi_inv == inf:
                        continue
                    try:
                        pi_inv += coeff ** 2 / float(div.pi)
                    except ZeroDivisionError:
                        pi_inv = inf
                pi = 1. / pi_inv
                tau = pi * mu
                return var.update_message(self, pi, tau)

        class TruncateFactor(Factor):

            def __init__(self, var, v_func, w_func, draw_margin):
                super(TruncateFactor, self).__init__([var])
                self.v_func = v_func
                self.w_func = w_func
                self.draw_margin = draw_margin

            def up(self):
                val = self.var
                msg = self.var[self]
                div = val / msg
                sqrt_pi = sqrt(div.pi)
                args = (div.tau / sqrt_pi, self.draw_margin * sqrt_pi)
                v = self.v_func(*args)
                w = self.w_func(*args)
                denom = (1. - w)
                pi, tau = div.pi / denom, (div.tau + sqrt_pi * v) / denom
                return val.update_value(self, pi, tau)

        # Create variables
        # Variables are gaussians so we use rate
        rating_vars = [Variable() for x in flat_ratings]
        perf_vars = [Variable() for x in flat_ratings]
        team_perf_vars = [Variable() for x in sorted_ratings]
        team_diff_vars = [Variable() for x in sorted_ratings[:-1]]
        team_sizes = [len(t) for t in sorted_teams]
        team_sizes = [sum(team_sizes[0:i+1]) for i, _ in enumerate(team_sizes)]

        def v_win(diff, draw_margin):
            """The non-draw version of "V" function.  "V" calculates a
            variation of a mean.
            """
            x = diff - draw_margin
            denom = norm.cdf(x)
            return (norm.pdf(x) / denom) if denom else -x

        def v_draw(diff, draw_margin):
            """The draw version of "V" function."""
            abs_diff = abs(diff)
            a, b = draw_margin - abs_diff, -draw_margin - abs_diff
            denom = norm.cdf(a) - norm.cdf(b)
            numer = norm.pdf(b) - norm.pdf(a)
            return ((numer / denom) if denom else a) * (-1 if diff < 0 else +1)

        def w_win(diff, draw_margin):
            """The non-draw version of "W" function.  "W" calculates a
            variation of a standard deviation.
            """
            x = diff - draw_margin
            v = v_win(diff, draw_margin)
            w = v * (v + x)
            if 0 < w < 1:
                return w
            raise FloatingPointError()

        def w_draw(diff, draw_margin):
            """The draw version of "W" function."""
            abs_diff = abs(diff)
            a, b = draw_margin - abs_diff, -draw_margin - abs_diff
            denom = norm.cdf(a) - norm.cdf(b)
            if not denom:
                raise FloatingPointError()
            v = v_draw(abs_diff, draw_margin)
            return (v ** 2) + (a * norm.pdf(a) - b * norm.pdf(b)) / denom

        # ------ Build factor graph ------ #

        rating_layer = [PriorFactor(rating_var, rating, tau) for rating_var,
                        rating in zip(rating_vars, flat_ratings)]

        perf_layer = [LikelihoodFactor(rating_var, perf_var, beta ** 2) for
                      rating_var, perf_var in zip(rating_vars, perf_vars)]

        team_perf_layer = []
        for team, team_perf_var in enumerate(team_perf_vars):
            start = team_sizes[team-1] if team > 0 else 0
            end = team_sizes[team]

            child_perf_vars = perf_vars[start:end]
            coefficients = flat_weights[start:end]

            team_perf_layer.append(SumFactor(team_perf_var, child_perf_vars,
                                             coefficients))

        team_diff_layer = \
            [SumFactor(team_diff_var, team_perf_vars[team:team+2], [1, -1])
             for team, team_diff_var in enumerate(team_diff_vars)]

        trunc_layer = []
        for x, team_diff_var in enumerate(team_diff_vars):
            # TODO: Make if statement for dynamic draw probability
            size = sum([len(x) for x in sorted_ratings[x:x+2]])
            draw_margin = norm.ppf((draw_probability + 1) / 2.) \
                * sqrt(size) * beta
            if ranks[x] == ranks[x + 1]:  # if tie
                v_func, w_func = v_draw, w_draw
            else:
                v_func, w_func = v_win, w_win
            trunc_layer.append(TruncateFactor(team_diff_var, v_func,
                                              w_func, draw_margin))

        """layers = [rating_layer, perf_layer, team_perf_layer,
                  team_diff_layer, trunc_layer]"""

        # ------ Iterative algorithm ------ #

        for factor in rating_layer + perf_layer + team_perf_layer:
            factor.down()

        team_diff_len = len(team_diff_layer)

        for iteration in range(10):
            if team_diff_len == 1:
                # only two teams
                team_diff_layer[0].down()
                delta = trunc_layer[0].up()
            else:
                # multiple teams
                delta = 0
                for iteration in range(team_diff_len - 1):
                    team_diff_layer[iteration].down()
                    delta = max(delta, trunc_layer[iteration].up())
                    team_diff_layer[iteration].up(1)  # up to right variable
                for iteration in range(team_diff_len - 1, 0, -1):
                    team_diff_layer[iteration].down()
                    delta = max(delta, trunc_layer[iteration].up())
                    team_diff_layer[iteration].up(0)  # up to left variable

            if delta <= tolerance:
                break
        # up both ends
        team_diff_layer[0].up(0)
        team_diff_layer[team_diff_len - 1].up(1)
        # up the remainder of the black arrows
        for factor in team_perf_layer:
            for iteration in range(len(factor.vars) - 1):
                factor.up(iteration)
        for factor in perf_layer:
            factor.up()

        # ------ Update the new ratings ------ #

        p = 0
        for team, team_ratings in enumerate(sorted_ratings):
            for individual, individual_rating in enumerate(team_ratings):
                sorted_ratings[team][individual] = \
                    Rate(rating_layer[p].var.mu, rating_layer[p].var.sigma)
                p += 1

        # Restore original ordering
    return new_ratings


def trueskill2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]"
) -> "list[Rate]":
    raise NotImplementedError()
