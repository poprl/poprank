import unittest
import functools

from poprank.core import Game, Player
from poprank.leagues import RankedPopulation
from poprank.ratings import win_draw_loss_rating


class TestRankedPopulation(unittest.TestCase):

    def setUp(self) -> None:
        self._spanish_league = RankedPopulation(
            rating_fn=functools.partial(
                win_draw_loss_rating,
                win_value=3,
                draw_value=1,
                loss_value=0
            )
        )
        self._players_id = ['Real_Madrid', 'FC_Barcelona']
        self._players = [
            Player(player_id, 0) for player_id in self._players_id]

        for player in self._players:
            self._spanish_league.add(
                player=player
            )

    def tearDown(self) -> None:
        self._spanish_league = None

    def test_initial_rank_preserves_insertion(self):
        rank = self._spanish_league.rank()
        self.assertEqual(rank, self._players)

    def test_rank_after_one_game(self):
        self._spanish_league.update(
            game=Game(
                players=self._players,
                scores=[1, 2]  # Real Madrid 1 - Barcelona 2
            )
        )
        rank = self._spanish_league.rank()
        self.assertEqual(rank, self._players[::-1])

    def test_rank_remains_unaltered_after_ties(self):
        games = [
            Game(
                players=self._players,
                scores=[1, 1]  # Real Madrid 1 - Barcelona 1
            ),
            Game(
                players=self._players,
                scores=[2, 2]  # Real Madrid 1 - Barcelona 1
            ),
            Game(
                players=self._players,
                scores=[2, 2]  # Real Madrid 1 - Barcelona 1
            ),
        ]

        for game in games:
            self._spanish_league.update(game)

        rank = self._spanish_league.rank()
        self.assertEqual(list(rank), self._players)
