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
    """
    base: float = 10  # the 10 in 10**(RA/400)
    spread: float = 400  # the 400 in 10**(RA/400)

    def expected_outcome(self, opponent_elo: Rate) -> float:
        return 1/(1+self.base**((self.mu - opponent_elo.mu)/400))
