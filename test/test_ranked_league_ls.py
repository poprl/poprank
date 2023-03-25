import functools
import json
import unittest

from poprank.core import Game, Player
from poprank.leagues.ranked import RankedPopulation
from poprank.ratings.score_based import win_draw_loss_rating

COMPETITIONS = {
    "la_liga_2019": {
        "clubs_file": "test/fixtures/2019/es.1.clubs.json",
        "games_file": "test/fixtures/2019/es.1.json",
        "standings_file": "test/fixtures/2019/es.1.final.json"
    },
    "premier_league_2019": {
        "clubs_file": "test/fixtures/2019/en.1.clubs.json",
        "games_file": "test/fixtures/2019/en.1.json",
        "standings_file": "test/fixtures/2019/en.1.final.json"
    }
}


class TestRankedPopulationLargeScale(unittest.TestCase):

    def _setup_competition(
            self, clubs_file: str, games_file: str, standings_file: str):
        with open(clubs_file) as f:
            data = json.load(f)
            clubs = data['clubs']

        self._players = {
           club['name']: Player(id=club['name'], rating=0) for club in clubs
        }

        with open(games_file) as f:
            games = json.load(f)

        self._games = [
            Game(
                players=[
                    self._players[game['team1']],
                    self._players[game['team2']]
                ],
                scores=game['score']['ft']
            ) for game in games['matches'] if 'score' in game
        ]

        with open(standings_file) as f:
            results = json.load(f)

        self._results = {
            club['id']: Player(**club) for club in results['clubs']
        }

    def setUp(self) -> None:
        self._players = None
        self._games = None
        self._results = None

    def _simulate_league_and_rank(self):
        league = RankedPopulation(
            rating_fn=functools.partial(
                win_draw_loss_rating,
                win_value=3,
                draw_value=1,
                loss_value=0
            )
        )

        for player_id, player in self._players.items():
            league.add(player)

        for game in self._games:
            league.update(
                game
            )

        return league.rank()

    def test_la_liga_2019_final_league_ranking(self):
        self._setup_competition(**COMPETITIONS['la_liga_2019'])

        rank = self._simulate_league_and_rank()

        self.assertEqual(rank, sorted(self._results.values(), reverse=True))

    def test_premier_league_2019_final_league_ranking(self):
        self._setup_competition(**COMPETITIONS['premier_league_2019'])

        rank = self._simulate_league_and_rank()

        self.assertEqual(rank, sorted(self._results.values(), reverse=True))
