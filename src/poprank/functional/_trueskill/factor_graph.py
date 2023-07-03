"""
Caution
=======

The TrueSkill part of this project is opened under the BSD license but the
TrueSkill(TM) brand is not. Microsoft permits only Xbox Live games or
non-commercial projects to use TrueSkill(TM). If your project is
commercial, you should find another rating system.

References
==========

The core ideas used in this project were described in
"TrueSkill (TM): A Bayesian Skill Rating System" available at
http://research.microsoft.com/apps/pubs/default.aspx?id=67956

Some concepts were based on Jeff Moser's code and documents, available
at http://www.moserware.com/2010/03/computing-your-skill.html

The code in this file was adapted from https://github.com/sublee/trueskill,
where this licence comes from.

License
=======

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
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

from dataclasses import dataclass
from math import sqrt
from scipy.stats import norm
from typing import Callable
from poprank import Rate

INF: float = float("inf")


@dataclass
class Gaussian(Rate):
    """Alternative representation of a gaussian

    Attributes:
        pi (float): Precision, the inverse of the variance
        tau (float): Precision adjusted mean: precision times mean"""
    __pi: float
    __tau: float

    def __init__(self, mu: float = None, std: float = None,
                 pi: float = 0., tau: float = 0.):
        if mu is not None:  # Note: sigma should be nonzero
            pi = std ** -2
            tau = pi * mu
        self.__pi = pi
        self.__tau = tau

    @property
    def pi(self) -> float:
        return self.__pi

    @pi.setter
    def pi(self, value) -> None:
        self.__pi = value

    @property
    def tau(self) -> float:
        return self.__tau

    @tau.setter
    def tau(self, value) -> None:
        self.__tau = value

    @property
    def mu(self) -> float:
        """A property which returns the mean."""
        return self.pi and self.tau / self.pi

    @mu.setter
    def mu(self, value) -> None:
        self.__tau = self.__pi * value

    @property
    def std(self) -> float:
        return sqrt(1. / self.__pi)

    @std.setter
    def std(self, value) -> None:
        self.__tau /= self.__pi
        self.__pi = 1. / value ** 2
        self.__tau *= self.__pi

    def __mul__(self, other: "Gaussian") -> "Gaussian":
        """Multiplication between two Gaussians"""
        pi, tau = self.pi + other.pi, self.tau + other.tau
        return Gaussian(pi=pi, tau=tau)

    def __truediv__(self, other: "Gaussian") -> "Gaussian":
        """Division between two Gaussians"""
        pi, tau = self.pi - other.pi, self.tau - other.tau
        return Gaussian(pi=pi, tau=tau)


class Variable(Gaussian):
    """A variable in the factor graph. Inherits from Gaussian.

    Attributes:
        messages (dict["Factor", Gaussian]): This variable's messages

    Methods:
        set(self, value: Gaussian) -> float: _description_
        delta(self, other: Gaussian) -> float: _description_
        update_message(self, factor: "Factor", pi: float = 0.,
                        tau: float = 0) -> float: _description_
        update_value(self, factor: "Factor", pi: float = 0,
                        tau: float = 0, value: Gaussian = None) -> float:
            _description_
    """

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
    """A factor in the factor graph"""

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


def flatten(array: list) -> list:
    """return a flattened copy of an array"""
    new_array = []
    for x in array:
        if isinstance(x, list):
            new_array.extend(flatten(x))
        else:
            new_array.append(x)
    return new_array
