import unittest
import json
from random import choices
from string import ascii_letters
from os.path import dirname
from popcore import Interaction, Team
from poprank import Rate
from poprank.functional.trueskill import trueskill

PRECISION = 5


class TestTrueskillFunctional(unittest.TestCase):
    def test_trueskill_win(self) -> None:
        """Default single interaction win case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [1, 0])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [[Rate(29.39583201999916, 7.171475587326195)],
                            [Rate(20.604167980000835, 7.171475587326195)]]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in g_results],
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in expected_results])

    def test_trueskill_draw(self) -> None:
        """Default single interaction draw case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [.5, .5])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [[Rate(25, 6.457519662317322)],
                            [Rate(25, 6.457519662317322)]]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in g_results],
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in expected_results])

    def test_trueskill_loss(self) -> None:
        """Default single interaction loss case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [0, 1])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [[Rate(20.604167980000835, 7.171475587326195)],
                            [Rate(29.39583201999916, 7.171475587326195)]]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [[Rate(round(x.mu, 3), round(x.std, 3)) for x in y]
             for y in g_results],
            [[Rate(round(x.mu, 3), round(x.std, 3))
             for x in y] for y in expected_results])

    def test_trueskill_complex_interaction(self) -> None:
        "Complicated interaction involving many teams of different sizes"
        players = [Team(name="1", members=["a", "b"]),
                   "c",
                   Team(name="2", members=["d", "e", "f"]),
                   Team(name="3", members=["g", "h"])]
        interactions = [Interaction(["1", "c", "2", "3"], [1, 2, 2, 3])]
        ratings = [[Rate(25, 25/3), Rate(25, 25/3)],
                   Rate(25, 25/3),
                   [Rate(29, 25/3), Rate(25, 8), Rate(20, 25/3)],
                   [Rate(25, 25/3), Rate(25, 25/3)]]
        expected_results = [[Rate(17.985454702066367, 7.249488484808368),
                             Rate(17.985454702066367, 7.249488484808368)],
                            [Rate(38.18810609223555, 6.503173849859338)],
                            [Rate(20.166629629774945, 7.337190164950848),
                             Rate(16.859096620101557, 7.123373401310813),
                             Rate(11.166629629774949, 7.337190164950848)],
                            [Rate(27.659809575923095, 7.596444468793693),
                             Rate(27.659809575923095, 7.596444468793693)]]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in g_results],
            [[Rate(round(x.mu, PRECISION), round(x.std, PRECISION)) for x in y]
             for y in expected_results])

    def test_trueskill_full_scale(self):
        d: str = dirname(__file__)
        games_filepath: str = f"{d}/fixtures/trueskill_tournament.json"
        with open(games_filepath, 'r') as f:
            data = json.load(f)

        interactions = [Interaction(
            players=[Team(name="".join(choices(ascii_letters, k=10)),
                          members=t) for t in i["players"]],
            outcomes=i["outcomes"]) for i in data["interactions"]]

        ratings = [Rate(25, 25/3) for x in data["players"]]

        results = trueskill(data["players"], interactions, ratings)

        expected_results = [Rate(51.90633032568547, 5.152291913349492),
                            Rate(-0.14687999106540617, 0.9365996379522524),
                            Rate(0.4540650480905728, 0.9311947030252915),
                            Rate(0.13313566468497742, 0.9304144846514893),
                            Rate(-0.7736780130962938, 0.92086459345153),
                            Rate(-0.5018216589497285, 0.9485525720071926),
                            Rate(-0.35077741217085373, 0.9288505181979116),
                            Rate(0.28311953890041597, 0.9378992714141935),
                            Rate(1.0882886073107747, 0.916306838738594),
                            Rate(0.5443461596913228, 0.9357279979765744),
                            Rate(-0.6461492028820802, 0.9440604773084467),
                            Rate(0.7033417455527605, 0.9331052344504057),
                            Rate(0.6610986541549926, 0.9289346121974258),
                            Rate(-0.36342655962184184, 0.9265334151336572),
                            Rate(0.48616705037407654, 0.9210727959787768),
                            Rate(0.6345274056468461, 0.9272344877371177),
                            Rate(0.049530589203413064, 0.9351551557875168),
                            Rate(-0.15527817402809876, 0.9295633594749055),
                            Rate(0.9121976600444945, 0.9251203660803101),
                            Rate(-0.013125637276157424, 0.9357784587476502),
                            Rate(0.6659428284211067, 0.925521673005709),
                            Rate(-0.2824424589989811, 0.9290250730425014),
                            Rate(-0.4629376609652222, 0.9235047606217096),
                            Rate(-0.3923957883947142, 0.9255975163061033),
                            Rate(-0.5923678314894874, 0.9015455536433515),
                            Rate(0.25030297875112184, 0.916379283032207)]

        self.assertListEqual()
