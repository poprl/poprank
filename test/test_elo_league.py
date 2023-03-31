import functools
import unittest
import numpy as np

from poprank.core import Game, Player
from poprank.leagues.ranked import RankedPopulation
from poprank.ratings.score_based import elo_rating


class TestEloPopulation(unittest.TestCase):

    def _setup_one_game(self, win):

        clubs = ["A", "B"]
        init_elo = [2400, 2000]

        if win:
            results = [[4, 1]]
        else:
            results = [[1, 4]]

        self._players = {}

        for i in range(len(clubs)):
            club = clubs[i]
            i_elo = init_elo[i]
            self._players[club] = Player(id=club, rating=i_elo)

        self._games = []
        for i in range(len(results)):
            self._games.append(
                Game(
                    players=[
                        self._players["A"],
                        self._players[clubs[i+1]],
                    ],
                    scores=results[i]
                )
            )

    def _setup_five_games(self):

        clubs = ["A", "B", "C", "D", "E", "F"]
        init_elo = [1613, 1609, 1477, 1388, 1586, 1720]
        results = [[1, 2], [2, 2], [3, 2], [3, 2], [1, 2]]

        self._players = {}

        for i in range(len(clubs)):
            club = clubs[i]
            i_elo = init_elo[i]
            self._players[club] = Player(id=club, rating=i_elo)

        self._games = []
        for i in range(len(results)):
            self._games.append(
                Game(
                    players=[
                        self._players["A"],
                        self._players[clubs[i+1]],
                    ],
                    scores=results[i]
                )
            )

    def _simulate_league_and_rank(self):
        league = RankedPopulation(
            rating_fn=functools.partial(
                elo_rating,
                k=32,
                rating_diff=400
            )
        )

        for player_id, player in self._players.items():
            league.add(player)

        for game in self._games:
            league.update(
                game
            )

        return [self._players[p].rating for p in self._players]

    def test_elo_after_one_win(self):
        self._setup_one_game(win=True)

        ratings = np.round(self._simulate_league_and_rank())

        self.assertEqual([2403, 1997], list(ratings))

    def test_elo_after_one_loss(self):
        self._setup_one_game(win=False)

        ratings = np.round(self._simulate_league_and_rank())

        self.assertEqual([2371, 2029], list(ratings))
