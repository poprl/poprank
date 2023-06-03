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
