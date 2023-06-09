import unittest
import json
from os.path import dirname
from popcore import Interaction
from poprank import EloRate
from poprank.functional.elo import bayeselo


class TestBayeseloFunctional(unittest.TestCase):

    def translateoutcome(self, outcome: str):
        if outcome == "1-0":
            return (1, 0)
        if outcome == "0-1":
            return (0, 1)
        return (.5, .5)

    def test_implementation_against_bayeselo(self):
        """Results BayesElo gives for this file
        fixtures/shortened_games.pgn
        1 Hiarcs 11.1         215   23   23   700   63%   120   26%
        2 Hiarcs 11           201   22   22   800   64%    96   25%
        3 Shredder 10         200   19   19   999   61%   116   22%
        4 Loop for Chess960   186   18   18  1100   61%    98   26%
        5 Hiarcs X54          179   22   22   800   64%    70   24%
        6 Spike 1.2 Turin     155   18   18  1100   57%   101   26%
        7 Fruit 2.2.1         152   18   18  1100   56%   101   24%
        8 Naum 2.1            108   19   19  1000   51%    98   25%
        9 Glaurung 1.2.1       71   18   18  1199   47%    88   23%
        10 Pharaon 3.5.1       -29   17   17  1500   43%    24   19%
        11 Ufim 8.02           -78   17   17  1500   41%    -9   17%
        12 Movei 00.8.383      -94   19   19  1200   43%   -32   18%
        13 Movei 00.8.366     -136   26   26   600   53%  -158   17%
        14 Hermann 1.9        -216   28   28   500   44%  -166   17%
        15 Hermann 1.7        -283   27   27   600   35%  -161   16%
        16 Aice 0.99.2        -303   25   25   700   33%  -166   15%
        17 Ayito 0.2.994      -328   27   27   600   32%  -185   12%"""

        d: str = dirname(__file__)
        games_filepath: str = f"{d}/fixtures/shortened_games.json"
        with open(games_filepath, "r") as f:
            games = json.load(f)

        self.assertEquals(len(games), 7999)  # Sanity check

        players = set()
        interactions = []
        for x in games:
            players.add(x[0])
            players.add(x[1])
            interactions.append(
                Interaction(players=[x[0], x[1]],
                            outcomes=self.translateoutcome(x[2])))

        elos = [EloRate(mu=0., std=0.) for x in players]

        actual_elos = [EloRate(x, 0) for x in
                       [215, 201, 200, 186, 179, 155, 152, 108, 71, -29, -78,
                        -94, -136, -216, -283, -303, -328]]
        results = bayeselo(players, interactions, elos)
        results.sort(reverse=True)
        self.assertListEqual(
            results,  # .sort(reverse=True),
            actual_elos)

    def test_win(self):
        players = ["a", "b"]
        interactions = [Interaction(players=players, outcomes=(1, 0))]
        elos = [EloRate(0., 0.) for x in players]
        results = bayeselo(players, interactions, elos)
        expected_results = [41, -41]
        self.assertListEqual(expected_results,
                             [round(x.mu) for x in results])

    def test_draw(self):
        players = ["a", "b"]
        interactions = [Interaction(players=players, outcomes=(.5, .5))]
        elos = [EloRate(0., 0.) for x in players]
        results = bayeselo(players, interactions, elos)
        expected_results = [-5, 5]
        self.assertListEqual(expected_results,
                             [round(x.mu) for x in results])

    def test_loss(self):
        players = ["a", "b"]
        interactions = [Interaction(players=players, outcomes=(0, 1))]
        elos = [EloRate(0., 0.) for x in players]
        results = bayeselo(players, interactions, elos)
        expected_results = [-48, 48]
        self.assertListEqual(expected_results,
                             [round(x.mu) for x in results])
