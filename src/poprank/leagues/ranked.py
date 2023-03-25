from typing import List
from poprank.core import (
    Game, Player, Population, RankingFn
)


class RankedPopulation(Population):

    def __init__(self, rating_fn: RankingFn) -> None:
        super().__init__(rating_fn)
        self._players = []

    def add(self, player: Player):
        self._players.append(player)

    def remove(self, player: Player):
        self._players.remove(player)

    def update(self, game: Game):
        ratings = self._rating_fn(game)
        for player, rating in zip(game.players, ratings):
            player.rating = rating

    def rank(self) -> List[Player]:
        return sorted(self._players, reverse=True)
