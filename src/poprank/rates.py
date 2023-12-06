from dataclasses import dataclass
from math import sqrt, log, pi, e
from statistics import NormalDist
import random


from abc import (
    ABC, abstractmethod
)
from typing import Any
import numpy as np


def _sigmoid(x: float, base: float, spread: float) -> float:
    return (1.0 + base ** (x / spread)) ** -1


@dataclass
class Rate:
    """Default Rate. It is the Canonical representation of a gaussian, where
    the mean is the rating and the standard deviation the uncertainty.

    :parameter float mu: Mean. Defaults to 0.

    :parameter float std: Standard deviation. Defaults to 1.
    """

    __mu: float
    __std: float

    def __init__(self, mu: float = 0, std: float = 1):
        self.__mu = mu
        self.__std = std

    def sample(self) -> float:
        raise NotImplementedError()

    def __lt__(self, other: 'Rate') -> bool:
        # TODO: is this right?
        return self.mu < other.mu

    def __eq__(self, other: 'Rate') -> bool:
        return np.isclose(self.mu, other.mu) and np.isclose(self.std, other.std)

    @property
    def mu(self) -> float:
        """
        Mean
        """
        return self.__mu

    @mu.setter
    def mu(self, value) -> None:
        self.__mu = value

    @property
    def std(self) -> float:
        """
        Standard deviation
        """
        return self.__std

    @std.setter
    def std(self, value) -> None:
        self.__std = value

    @abstractmethod
    def expected_outcome(self, opponent: "Rate"):
        """Probability that the player rate is greater than the opponent's rate
        given both distributions

        :param Rate opponent: Opponent's rating
        :return: The probability P(self>opponent).
        :rtype: float
        """
        mean = self.mu - opponent.mu
        standard_dev = sqrt(self.std ** 2 + opponent.std ** 2)
        return 1.0 - NormalDist(mean, standard_dev).cdf(x=0)


class RateModule(ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        *args: Any,
        **kwds: Any
    ) -> Any:
        self._rate()

    @abstractmethod
    def _rate(self):
        raise NotImplementedError()


class EloRate(Rate):
    """Elo rating.

    :param float base: base of the exponent in the elo formula
    :param float spread: divisor of the exponent in the elo formula

    See also :meth:`poprank.functional.elo.elo()`,
    :meth:`poprank.functional.bayeselo.bayeselo()`
    """
    # TODO: base and spread should be parameters.

    base: float = 10.  # the 10 in 10**(RA/400)
    spread: float = 400.  # the 400 in 10**(RA/400)

    def expected_outcome(self, opponent_elo: "EloRate") -> float:
        """Return the expected score against an opponent of the specified elo

        Uses the elo formula with self.base and self.spread substituted

        :parameter opponent_elo: (Rate) the elo of the opponent
        :return: The expected score.
        :rtype: float"""
        if not isinstance(opponent_elo, EloRate):
            raise TypeError("opponent_elo should be of type EloRate")
        skill_difference = opponent_elo.mu - self.mu
        # return 1.0 / (1.0 + self.base**(skill_difference / self.spread))
        return _sigmoid(skill_difference, base=self.base, spread=self.spread)

    @property
    def q(self):
        return log(self.base) / self.spread


class GlickoRate(EloRate):
    """Glicko rating

    :param float mu: Player's initial rating. Defaults to 1500.
    :param float std: Player's default standard deviation. Defaults to 350
    """

    time_since_last_competition: int = 0

    def __init__(self, mu: float = 1500, std: float = 350):
        Rate.__init__(self, mu, std)

    def reduce_impact(self, RD_i: float) -> float:
        """Originally g(RDi), reduces the impact of a game based on the
        opponent's rating_deviation


        :param float RD_i: Rating deviation of the opponent
        :param float q: Q constant. Typically ln(10)/400 in glicko1
            but equal to 1 for glicko2
        :return: g(RDi)
        :rtype: float
        """
        return 1 / sqrt(1 + (3 * (self.q**2) * (RD_i**2)) / (pi**2))

    def expected_outcome(self, opponent_glicko: "GlickoRate") -> float:
        """Calculate the expected outcome of a match in the glicko1 system

        :param GlickoRate opponent_glicko: Opponent's rating
        :return: The expected score.
        :rtype: float"""
        if not isinstance(opponent_glicko, GlickoRate):
            raise TypeError("opponent_glicko should be of type Glicko1Rate")

        # g_RD_i on the Glicko paper
        impact_scale = self.reduce_impact(opponent_glicko.std)
        skill_difference = opponent_glicko.mu - self.mu

        return _sigmoid(
            impact_scale * skill_difference, self.base, self.spread)


class Glicko2Rate(GlickoRate):
    """Glicko2 rating.

    :param float mu: Player's initial rating. Defaults to 0.
    :param float std: Player's default standard deviation. Defaults to 1
    :param float base: The base of the exponent in the elo formula.
    Defaults to 10.0.
    :param float spread:The divisor of the exponent in the elo formula.
    Defaults to 400.0.
    """
    base: float = e
    spread: float = 1.0
    time_since_last_competition: int = 0
    volatility: float = 0.06

    def __init__(self, mu: float = 0, std: float = 1,
                 base: float = 10.0, spread: float = 400.0):
        Rate.__init__(self, mu, std)
        self.base = base
        self.spread = spread


class TrueSkillRate(Rate):
    """TrueSkill rating.

    :param float mu: Player's initial rating. Defaults to 25.
    :param float std: Player's default standard deviation. Defaults to 25/3
    """
    def __init__(self, mu: float = 25, std: float = 25/3):
        Rate.__init__(self, mu, std)


class MeloRate(Rate):
    """mElo2k rating.

    :param float mu: Player's initial rating. Defaults to 0.
    :param float std: Player's default standard deviation. Defaults to 1
    :param int k: The mElo rating will have 2k dimensions. Defaults to 1.
    :param list vector: The initial mElo vector. Should be of length 2k. If
        None, it will be initialized to a uniform[-0.5, 0.5] random vector of
        length 2k. Defaults to None.
    """
    # TODO: Test behavior for k = 0
    def __init__(self, mu: float, std: float, k: int = 1,
                 vector: None | list = None):
        Rate.__init__(self, mu, std)
        if vector is None:
            self.vector = [random.random()-0.5 for x in range(2*k)]
        else:
            assert len(vector) == 2*k, "The vector must be of length 2k"
            self.vector = vector
        self.k = k

    def _build_omega(self, k):
        omega = [[0 for x in range(2*k)] for y in range(2*k)]
        for i in range(k):
            omega[2*i][2*i+1] = 1
            omega[2*i+1][2*i] = -1
        return omega

    def expected_outcome(self, opponent: "MeloRate") -> float:
        """Expected score of the player against an opponent with the specified
        rating.

        :param MeloRate opponent: mElo2k Rate of the opponent. K must be the
            same for both players.

        :return: The expected score.
        :rtype: float
        """
        omega = self._build_omega(self.k)
        adjustment = [sum([i * j for i, j in zip(self.vector, omega[a])]) for a in range(self.k*2)]
        adjustment = sum([i*j for i, j in zip(adjustment, opponent.vector)])
        return _sigmoid(self.mu - opponent.mu + adjustment, base=e, spread=1.0)
