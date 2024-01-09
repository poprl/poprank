import unittest
import numpy as np
from popcore import History, Interaction

from poprank import Rate
from poprank.functional.rates import laplacian


class TestLaplacianRating(unittest.TestCase):
    def test_rock_paper_scissor(self):
        interactions = [
            Interaction(["s", "s"], [0.5, 0.5]),
            Interaction(["s", "r"], [0.0, 1.0]),
            Interaction(["s", "p"], [1.0, 0.0]),
            Interaction(["r", "r"], [0.5, 0.5]),
            Interaction(["r", "p"], [0.0, 1.0]),
            Interaction(["r", "s"], [1.0, 0.0]),
            Interaction(["p", "p"], [0.5, 0.5]),
            Interaction(["p", "s"], [0.0, 1.0]),
            Interaction(["p", "r"], [1.0, 0.0]),
        ]

        history = History.from_interactions(interactions)

        rates = [Rate(0, 1) for p in history.players]

        ratings = laplacian(None, history._interactions, rates)

        for player, rating in zip(['R', 'P', 'S'], ratings):
            print(f"{player}: {rating}")

        assert np.allclose(
            ratings - ratings.max(),
            np.zeros_like(ratings),
            atol=1e-7
        )  # desc: B, A, C

    def test_lanctot_et_al(self):
        """
        
        """
        ratings = laplacian(
            np.array([
                [0.0, 0.2, 0.6],
                [0.8, 0.0, 0.6],
                [0.4, 0.4, 0.0]
            ])
        )

        for player, rating in zip(['A', 'B', 'C'], ratings):
            print(f"{player}: {rating}")

        self.assertTrue(
            np.array_equal(np.argsort(ratings), [1, 0, 2])
        )  # desc: B, A, C

    def test_lanctot_et_al_condorcet_margin(self):
        """

        """
        ratings = laplacian(
            np.array([
                [0.0, 1.0, 3.0],
                [4.0, 0.0, 3.0],
                [2.0, 2.0, 0.0]
            ])
        )

        for player, rating in zip(['A', 'B', 'C'], ratings):
            print(f"{player}: {rating}")

        self.assertTrue(
            np.array_equal(np.argsort(ratings), [1, 0, 2])
        )  
        assert True

