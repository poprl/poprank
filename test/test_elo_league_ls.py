import functools
import unittest
import pandas as pd

from poprank.core import Game, Player
from poprank.leagues.ranked import RankedPopulation
from poprank.ratings.score_based import elo_rating


COMPETITIONS = {
    "nba_2015": {
        "games_file": "test/fixtures/nba/nba_2015_season.csv",
        "standings_file": "test/fixtures/nba/nba_2015_standings.csv"
    }
}


class TestEloPopulationLargeScale(unittest.TestCase):

    def _setup_competition(
            self, games_file: str, standings_file: str):

        games = pd.read_csv(games_file)
        clubs = list(games["team_id"].unique())

        self._players = {
            club: Player(id=club, rating=0) for club in clubs
        }

        self._games = []
        for index, game in games.iterrows():
            self._games.append(
                Game(
                    players=[
                        self._players[game["team_id"]],
                        self._players[game["opp_id"]],
                    ],
                    scores=[game["pts"], game["opp_pts"]]
                )
            )

        results = pd.read_csv(standings_file)

        self._results = {}
        for idx, standing in results.iterrows():
            self._results[standing["team_id"]] = Player(
                id=standing["team_id"], rating=standing["elo_n"])

    def setUp(self) -> None:
        self._players = None
        self._games = None
        self._results = None

    def _simulate_league_and_rank(self):
        league = RankedPopulation(
            rating_fn=functools.partial(
                elo_rating,
                k=32,
            )
        )

        for player_id, player in self._players.items():
            league.add(player)

        for game in self._games:
            league.update(
                game
            )

        return league.rank()

    def test_nba_final_elo(self):
        self._setup_competition(**COMPETITIONS['nba_2015'])
 
        rank = self._simulate_league_and_rank()

        true_sorted_teams = sorted(self._results.values(), reverse=True)

        true_sorted_teams = [team.id for team in true_sorted_teams]
        elo_ranked_teams = [team.id for team in rank]

        self.assertEqual(elo_ranked_teams, true_sorted_teams)
