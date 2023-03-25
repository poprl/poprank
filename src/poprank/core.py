import abc
from dataclasses import dataclass
from typing import Callable, List, Union


@dataclass
class Player:
    id: str
    rating: Union[int, float]

    def __lt__(self, other):
        return self.rating < other.rating


@dataclass
class Game:
    """_summary_
        players: players involved in the game
        results: one score for each player involved in the game
    """
    players: List[Player]
    scores: List[float]


RankingFn = Callable[[Game], List[float]]


class Population(abc.ABC):

    def __init__(self, rating_fn: RankingFn) -> None:
        super().__init__()
        self._rating_fn = rating_fn

    @abc.abstractmethod
    def add(self, player: Player):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove(self, player: Player):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, game: Game):
        raise NotImplementedError()

    @abc.abstractmethod
    def rank(self) -> List[Player]:
        raise NotImplementedError()
