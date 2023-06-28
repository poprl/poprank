
from scipy.stats import norm
from dataclasses import dataclass
from math import sqrt, log, pi, e
from abc import (
    ABC, abstractmethod
)
from typing import Any

INF: float = float("inf")


def _sigmoid(x: float, base: float, spread: float) -> float:
    return (1.0 + base ** (x / spread)) ** -1
 

@dataclass
class Rate:
    """Canonical representation of a gaussian

    Attributes:
        mu (float): Mean
        std (float): Standard deviation"""
    __mu: float
    __std: float

    def __init__(self, mu: float = 0, std: float = 1):
        self.__mu = mu
        self.__std = std

    def sample(self) -> float:
        pass

    @property
    def mu(self) -> float:
        return self.__mu

    @mu.setter
    def mu(self, value) -> None:
        self.__mu = value

    @property
    def std(self) -> float:
        return self.__std

    @std.setter
    def std(self, value) -> None:
        self.__std = value

    def expected_outcome(self, opponent: "Rate"):
        """probability that player rate > opponent rate given both
        distributions"""
        mean = self.mu - opponent.mu
        standard_dev = sqrt(self.std**2 + opponent.std**2)
        return 1 - norm.cdf(x=0, loc=mean, scale=standard_dev)


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
      
    @abstractmethod
    def expected_outcome(self, opponent_rate: 'Rate') -> float:
        raise NotImplementedError()

    def __lt__(self, other: 'Rate') -> bool:
        # TODO: is this right?
        return self.mu < other.mu


class RateModule(ABC):
    """_summary_

    Args:
        ABC (_type_): _description_
    """
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
    """Elo rating

    Args:
        base (float): base of the exponent in the elo for__mula
        spread (float): divisor of the exponent in the elo for__mula
    Methods:
        expected_outcome(opponent_elo: Rate) -> float: Return the probability
            of winning against an opponent of the specified elo
    """
    base: float = 10.  # the 10 in 10**(RA/400)
    spread: float = 400.  # the 400 in 10**(RA/400)

    def __init__(self, mu: float = 1000., std: float = 200,
                 base: float = 10., spread: float = 400.):
        self.base = base
        self.spread = spread
        super().__init__(mu=mu, std=std)

    def expected_outcome(self, opponent_elo: "EloRate") -> float:
        """Return the probability of winning against an opponent of the
        specified elo

        Uses the elo for__mula with self.base and self.spread substituted

        Args:
            opponent_elo (Rate): the elo of the opponent"""
        if not isinstance(opponent_elo, EloRate):
            raise TypeError("opponent_elo should be of type EloRate")
        skill_difference = opponent_elo.mu - self.mu
        # return 1.0 / (1.0 + self.base**(skill_difference / self.spread))
        return _sigmoid(skill_difference, base=self.base, spread=self.spread)

    @property
    def q(self):
        return log(self.base) / self.spread


class GlickoRate(EloRate):
    """Glicko rating"""

    time_since_last_competition: int = 0

    def reduce_impact(self, RD_i: float) -> float:
        """Originally g(RDi), reduced the impact of a game based on the
        opponent's rating_deviation

        Args:
            RD_i (float): Rating deviation of the opponent
            q (float): Q constant. Typically ln(10)/400 in glicko1
                but equal to 1 for glicko2
        """
        return 1 / sqrt(1 + (3 * (self.q**2) * (RD_i**2)) / (pi**2))

    def expected_outcome(self, opponent_glicko: "GlickoRate") -> float:
        """Calculate the expected outcome of a match in the glicko1 system"""
        if not isinstance(opponent_glicko, GlickoRate):
            raise TypeError("opponent_glicko should be of type Glicko1Rate")

        # g_RD_i on the Glicko paper
        impact_scale = self.reduce_impact(opponent_glicko.std)
        skill_difference = opponent_glicko.mu - self.mu

        return _sigmoid(
            impact_scale * skill_difference, self.base, self.spread)


class Glicko2Rate(GlickoRate):
    """Glicko rating"""
    base: float = e
    spread: float = 1.0
    time_since_last_competition: int = 0
    volatility: float = 0.06
