# core
from abc import ABC, abstractmethod
from math import sqrt
from statistics import NormalDist
from typing import Any, Generic, Iterable, List, TypeVar
# third-party
import numpy as np
# internal

from popcore.core import (
    Interaction, Player, Population,
)


class Rate:
    """
        Default Rate. It is the canonical representation of a gaussian, where
        the mean is the rating and the standard deviation the uncertainty.

        :param float mu: Mean. Defaults to 0.

        :param float std: Standard deviation. Defaults to 1.
    """

    __mu: float
    __std: float

    def __init__(self, mu: float = 0.0, std: float = 1.0):
        self.__mu = mu
        self.__std = std

    def sample(self) -> float:
        raise NotImplementedError()

    def __lt__(self, other: 'Rate') -> bool:
        # TODO: is this right? Assume it is for now
        return self.mu < other.mu

    def __eq__(self, other: 'Rate') -> bool:
        """
            TODO: Maybe use a metric (KL, TV?) between Gaussians?
            TODO: If left untouched, verify tolerance.
        """
        is_equal = np.isclose(self.mu, other.mu)
        is_equal &= np.isclose(self.std, other.std)
        return is_equal

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
    def std(self, value: float) -> None:
        self.__std = value

    def predict(self, opponent: "Rate"):
        """Probability that the player rate is greater than the opponent's rate
        given both distributions

        :param Rate opponent: Opponent's rating
        :return: The probability P(self>opponent).
        :rtype: float
        """
        mean = self.mu - opponent.mu
        standard_dev = sqrt(self.std ** 2 + opponent.std ** 2)
        return 1.0 - NormalDist(mean, standard_dev).cdf(x=0)


RateType = TypeVar("RateType", bound=Rate)


class RateModule(Generic[RateType], ABC):
    """
        A Rate Module contains a sequence of ratings.
    """
    def __init__(
        self, population: Population, rates: List[Rate] = None,
        default_rate: float = 1.0
    ):
        self._population = population
        if not rates:
            rates = self._defaults(default_rate)
        self._rates: List[RateType] = rates if rates else [rates]

    def _defaults(self, default_rate: float) -> List[RateType]:
        """
            Establishes the default rates for the players in the population.

        :param default_rate: Default rate value for every player
        :type default_rate: float
        :return: A list with every player default rate.
        :rtype: List[RateType]
        """
        return [Rate(default_rate) for _ in self._population.players]

    @abstractmethod
    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[RateType]:
        raise NotImplementedError()

    def __call__(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[RateType]:
        rates = self._rate(
            interactions, **kwds
        )
        self._rates.append(rates)
        return rates

    @property
    def rates(self) -> List[RateType]:
        return self._rates[-1]


class Rank:
    """
        zero-based rank

         Includes the `compose` and `inverse` operations from
         S_n the permutation group.
    """
    def __init__(self, idx: Iterable[int]) -> None:
        self._idx = np.asarray(idx)  # TODO: test the rank is complete
        self.n = len(self._idx)

    def compose(self, other: 'Rank') -> 'Rank':
        if self.n != other.n:
            raise ValueError()  # TODO: exception raising
        return Rank(other._idx[self._idx])

    def inverse(self) -> 'Rank':
        return Rank(np.argsort(self._idx))

    def __iter__(self) -> Iterable[int]:
        return iter(self._idx)

    def __mul__(self, other: 'Rank') -> 'Rank':
        return self.compose(other)

    def __pow__(self, value: int):
        if value != -1:
            raise ValueError()
        return self.inverse()

    def __repr__(self) -> str:
        seq = [str(idx) for idx in self._idx]
        return f"Rank([{','.join(seq)}])"

    def __eq__(self, other: 'Rank') -> bool:
        return np.array_equal(self._idx, other._idx)

    def __array__(self):
        return self._idx


class RankModule(Generic[RateType]):
    """
        Ranks a population of players by a comparable field.
    """
    def __init__(
        self,
        population: Population
    ) -> None:
        self._population = population
        self._ranks: List[Rank] = []

    def _rank(self, rates: List[RateType], **kwds) -> Rank:
        return Rank(np.argsort(rates))

    def __call__(self, rates: List[RateType], **kwds: Any) -> Any:
        rank = self._rank(rates, **kwds)
        self._ranks.append(rank)
        return rank

    # @property
    # def ranks(self) -> list[int]:
    #     """Returns each player rank

    #     :return: List of each player's position within the rank.
    #     :rtype: list[int]
    #     """
    #     if not self._rank:
    #         self.rerank(inplace=True)

    #     return self._ranks

    # @property
    # def order(self):
    #     return len(self._rates)

    # @property
    # def players(self) -> np.ndarray[Any, PlayerType]:
    #     """ Returns the population players ordered by rank.

    #     :return: _description_
    #     :rtype: _type_
    #     """
    #     return self._population.players[self.rank]

    # def rerank(
    #     self, inplace: bool = True
    # ) -> 'Rank':
    #     """Recomputes the players ranking in the population.


    #     :param order: _description_, defaults to None
    #     :type order: str, optional
    #     :param inplace: _description_, defaults to True
    #     :type inplace: bool, optional
    #     :return: _description_
    #     :rtype: Rank
    #     """
    #     if not inplace:
    #         # TODO: does not make sense with rates detached
    #         # from the population.
    #         rank = Rank(self._population, self._rates)
    #         rank.rerank()
    #         return rank

    #     self._rank = np.argsort(self._rates)
    #     return self

    # def __iter__(self) -> Iterable[int]:
    #     return self._rank

    # def __array__(self) -> Iterable:
    #     return self._rank
