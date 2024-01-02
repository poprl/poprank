# core
from math import sqrt
from statistics import NormalDist
# third-party
import numpy as np
# internal
from popcore import Interaction, Player


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
            TODO: Maybe use a metric between Gaussians?
            TODO: If left untouced, verify tolerance.
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
    def std(self, value) -> None:
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


class Rank:
    """
        _summary_
    """
    def __init__(
        self,
        players: list[str],
        ratings: list[Rate],
    ) -> None:
        """
            TODO: Documentation
        """
        self._players = np.array(players, dtype=str)
        self._ratings = np.array(ratings, dtype=Rate)
        self._rank = np.argsort(self._ratings)

    @property
    def rank(self) -> list[int]:
        """Returns each player rank

        :return: List of each player's position within the rank.
        :rtype: list[int]
        """
        return self._rank

    @property
    def players(self) -> list[str]:
        """Returns players ordered by rank

        :return: _description_
        :rtype: _type_
        """
        return self._players[self._rank]

    def __array__(self):
        return self._rank
