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
