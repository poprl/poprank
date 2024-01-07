import unittest
import numpy as np

from popcore import Population
from poprank.functional.rates import laplacian


class TestLaplacianRating(unittest.TestCase):
    def test_rock_paper_scissor(self):
        payoff = np.array([
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [1, 0.0, 0.5]
        ])

        population = Population[RatedPlayer](
            uid="rock+paper+scissor",
            players=[RatedPlayer(uid) for uid in ["R", "P", "S"]]
        )
            

        ratings = laplacian(payoff)

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

