import unittest
from popcore import Interaction
from poprank import GlickoRate, Glicko2Rate
from poprank.functional.glicko import glicko, glicko2


class TestGlickoFunctional(unittest.TestCase):
    def test_glicko_win(self) -> None:
        """Default single interaction win case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [1, 0])]
        ratings = [
            GlickoRate(1000, 350), GlickoRate(1000, 350)
        ]
        expected_results = [
            GlickoRate(1162.212, 290.231),
            GlickoRate(837.788, 290.231)
        ]
        g_results = glicko(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [GlickoRate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            expected_results
        )

    def test_glicko_lose(self) -> None:
        """Default single interaction lose case"""
        players = ("a", "b")
        interactions = [Interaction(["a", "b"], [0, 1])]
        ratings = [
            GlickoRate(1000, 350), GlickoRate(1000, 350)
        ]
        expected_results = [
            GlickoRate(837.788, 290.231), GlickoRate(1162.212, 290.231)
        ]
        g_results = glicko(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [GlickoRate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            expected_results)

    def test_glicko_draw(self) -> None:
        """Default single interaction draw case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [.5, .5])]
        ratings = [
            GlickoRate(1000, 350), GlickoRate(1000, 350)
        ]
        expected_results = [
            GlickoRate(1000.0, 290.231), GlickoRate(1000.0, 290.231)
        ]
        g_results = glicko(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [GlickoRate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            expected_results
        )

    def test_glicko_multiple_interactions(self):
        """Example from Glickman's paper
        http://www.glicko.net/glicko/glicko.pdf"""
        players = ["a", "b", "c", "d"]
        interactions = [
            Interaction(["a", "b"], [1, 0]),
            Interaction(["a", "c"], [0, 1]),
            Interaction(["a", "d"], [0, 1]),
            Interaction(["b", "c"], [0, 1]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [.5, .5])
        ]
        ratings = [
            GlickoRate(1500, 200), GlickoRate(1400, 30),
            GlickoRate(1550, 100), GlickoRate(1700, 300)
        ]

        g_results = glicko(players, interactions, ratings)
        g_results = [
            GlickoRate(round(x.mu, 3), round(x.std, 3))
            for x in g_results
        ]

        expected_results = [
            GlickoRate(1464.106, 151.399), GlickoRate(1396.046, 29.800),
            GlickoRate(1588.344, 92.598), GlickoRate(1742.969, 194.514)
        ]

        self.assertListEqual(g_results, expected_results)

    def test_glicko_no_interactions(self) -> None:
        """No interaction case"""
        players = ["a", "b"]
        interactions = []
        ratings = [
            GlickoRate(1000, 350), GlickoRate(1000, 350)
        ]
        expected_results = [
            GlickoRate(1000.0, 350), GlickoRate(1000.0, 350)
        ]

        g_results = glicko(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [GlickoRate(round(x.mu, 3), round(x.std, 3)) for x in g_results],
            expected_results)


class TestGlicko2Functional(unittest.TestCase):
    def round_glicko2(self, g_results) -> "list[Glicko2Rate]":
        g_volatilities = [x.volatility for x in g_results]
        g_results = [
            Glicko2Rate(round(x.mu, 3), round(x.std, 3))
            for x in g_results
        ]
        for x, _ in enumerate(g_results):
            g_results[x].volatility = round(g_volatilities[x], 10)

        return g_results

    def test_glicko2_win(self) -> None:
        """Default single interaction win case"""
        players = ["a", "b"]
        interactions = [
            Interaction(["a", "b"], [1, 0])
        ]
        ratings = [
            Glicko2Rate(1000, 350), Glicko2Rate(1000, 350)
        ]
        expected_results = [
            Glicko2Rate(1162.311, 290.319), Glicko2Rate(837.689, 290.319)
        ]
        expected_volatilities = [
            0.060000000000000005, 0.060000000000000005
        ]

        for x, _ in enumerate(expected_results):
            expected_results[x].volatility = \
                round(expected_volatilities[x], 10)
        tau = 0.5
        g_results = glicko2(players, interactions, ratings, tau)
        g_results = self.round_glicko2(g_results)

        self.assertListEqual(g_results, expected_results)

    def test_glicko2_lose(self) -> None:
        """Default single interaction lose case"""
        players = ("a", "b")
        interactions = [Interaction(["a", "b"], [0, 1])]
        ratings = [
            Glicko2Rate(1000, 350), Glicko2Rate(1000, 350)
        ]
        expected_results = [
            Glicko2Rate(837.689, 290.319), Glicko2Rate(1162.311, 290.319)
        ]
        expected_volatilities = [
            0.060000000000000005, 0.060000000000000005
        ]

        for x, _ in enumerate(expected_results):
            expected_results[x].volatility = \
                round(expected_volatilities[x], 10)
        tau = 0.5
        g_results = glicko2(players, interactions, ratings, tau)
        g_results = self.round_glicko2(g_results)

        self.assertListEqual(g_results, expected_results)

    def test_glicko2_draw(self) -> None:
        """Default single interaction draw case"""
        players = ["a", "b"]
        interactions = [
            Interaction(["a", "b"], [.5, .5])
        ]
        ratings = [
            Glicko2Rate(1000, 350), Glicko2Rate(1000, 350)
        ]
        expected_results = [
            Glicko2Rate(1000.0, 290.319), Glicko2Rate(1000.0, 290.319)
        ]
        expected_volatilities = [
            0.060000000000000005, 0.060000000000000005
        ]

        for x, _ in enumerate(expected_results):
            expected_results[x].volatility = \
                round(expected_volatilities[x], 10)
        tau = 0.5
        g_results = glicko2(players, interactions, ratings, tau)
        g_results = self.round_glicko2(g_results)

        self.assertListEqual(g_results, expected_results)

    def test_glicko2_multiple_interactions(self):
        """Example from Glickman's paper
        http://www.glicko.net/glicko/glicko2.pdf"""
        players = ["a", "b", "c", "d"]
        interactions = [
            Interaction(["a", "b"], [1, 0]),
            Interaction(["a", "c"], [0, 1]),
            Interaction(["a", "d"], [0, 1]),
            Interaction(["b", "c"], [0, 1]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [.5, .5])
        ]
        ratings = [
            Glicko2Rate(1500, 200), Glicko2Rate(1400, 30),
            Glicko2Rate(1550, 100), Glicko2Rate(1700, 300)]
        tau = 0.5

        g_results = glicko2(players, interactions, ratings, tau)
        g_results = self.round_glicko2(g_results)

        expected_results = [
            Glicko2Rate(1464.051, 151.517), Glicko2Rate(1395.575, 31.522),
            Glicko2Rate(1588.701, 93.027), Glicko2Rate(1742.991, 194.563)
        ]

        expected_volatilities = [
            0.0599959842864885, 0.06000183590775173,
            0.060000172066474, 0.060000000000000005
        ]

        for x, _ in enumerate(expected_results):
            expected_results[x].volatility = \
                round(expected_volatilities[x], 10)

        self.assertListEqual(g_results, expected_results)

    def test_glicko2_no_interactions(self) -> None:
        """No interactions case"""
        players = ["a", "b"]
        interactions = []
        ratings = [
            Glicko2Rate(1000, 350), Glicko2Rate(1000, 350)
        ]
        expected_results = [
            Glicko2Rate(1000.0, 350.155), Glicko2Rate(1000.0, 350.155)
        ]
        expected_volatilities = [
            0.06, 0.06
        ]

        for x, _ in enumerate(expected_results):
            expected_results[x].volatility = \
                round(expected_volatilities[x], 10)
        tau = 0.5
        g_results = glicko2(players, interactions, ratings, tau)
        g_results = self.round_glicko2(g_results)

        self.assertListEqual(g_results, expected_results)
