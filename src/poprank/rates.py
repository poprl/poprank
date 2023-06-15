from math import sqrt, log, exp, pi
from abc import (
    ABC, abstractmethod
)
from typing import Any, NamedTuple


class Rate(NamedTuple):
    mu: float
    std: float

    def sample(self) -> float:
        pass


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
        base (float): base of the exponent in the elo formula
        spread (float): divisor of the exponent in the elo formula
    Methods:
        expected_outcome(opponent_elo: Rate) -> float: Return the probability
            of winning against an opponent of the specified elo
    """
    base: float = 10.  # the 10 in 10**(RA/400)
    spread: float = 400.  # the 400 in 10**(RA/400)

    def expected_outcome(self, opponent_elo: "EloRate") -> float:
        """Return the probability of winning against an opponent of the
        specified elo

        Uses the elo formula with self.base and self.spread substituted

        Args:
            opponent_elo (Rate): the elo of the opponent"""
        if not isinstance(opponent_elo, EloRate):
            raise TypeError("opponent_elo should be of type EloRate")

        return 1./(1.+self.base**((opponent_elo.mu - self.mu)/self.spread))


class GlickoRate(EloRate):
    """Glicko rating"""
    time_since_last_competition: int = 0
    volatility: float = 0.06

    @staticmethod
    def reduce_impact(RD_i: float, q: float) -> float:
        """Originally g(RDi), reduced the impact of a game based on the
        opponent's rating_deviation

        Args:
            RD_i (float): Rating deviation of the opponent
            q (float): Q constant. Typically ln(10)/400 in glicko1
                but equal to 1 for glicko2
        """
        return 1 / sqrt(1 + (3 * (q**2) * (RD_i**2)) / (pi**2))

    def glicko1_expected_outcome(self, opponent_glicko: "GlickoRate"):
        """Calculate the expected outcome of a match in the glicko1 system"""
        g_RD_i = GlickoRate.reduce_impact(opponent_glicko.std, log(self.base) /
                                          self.spread)
        return 1 / (1 + self.base ** (g_RD_i * (self.mu - opponent_glicko.mu)
                                      / (-1 * self.spread)))

    def glicko2_expected_outcome(self, opponent_glicko: "GlickoRate"):
        """Calculate the expected outcome of a match in the glicko2 system"""
        return 1 / (1 + exp(-1 *
                            GlickoRate.reduce_impact(opponent_glicko.std, 1) *
                            (self.mu - opponent_glicko.mu)))
