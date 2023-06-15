import unittest
from popcore import Interaction, Team
from poprank import Rate
from poprank.functional.trueskill import trueskill


class TestTrueskillFunctional(unittest.TestCase):
    def test_trueskill_win(self) -> None:
        """Default single interaction win case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [1, 0])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [Rate(29.39583201999916, 7.171475587326195),
                            Rate(20.604167980000835, 7.171475587326195)]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in expected_results])

    def test_trueskill_draw(self) -> None:
        """Default single interaction draw case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [.5, .5])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [Rate(25, 6.457519662317322),
                            Rate(25, 6.457519662317322)]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in expected_results])

    def test_trueskill_loss(self) -> None:
        """Default single interaction loss case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [0, 1])]
        ratings = [Rate(25, 25/3), Rate(25, 25/3)]
        expected_results = [Rate(20.604167980000835, 7.171475587326195),
                            Rate(29.39583201999916, 7.171475587326195)]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in expected_results])

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
                            Rate(38.18810609223555, 6.503173849859338),
                            [Rate(20.166629629774945, 7.337190164950848),
                             Rate(16.859096620101557, 7.123373401310813),
                             Rate(11.166629629774949, 7.337190164950848)],
                            [Rate(27.659809575923095, 7.596444468793693),
                             Rate(27.659809575923095, 7.596444468793693)]]
        g_results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            [Rate(round(x.mu, 3), round(x.std, 3)) for x in expected_results])
