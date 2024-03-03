from itertools import chain
import unittest
import numpy as np
from popcore import Interaction

from poprank.functional.rates import laplacian


class TestLaplacianRating(unittest.TestCase):
    def _lanctot_et_al_interactions(self):
        return chain(
            [
                Interaction(["A", "B"], [1.0, 0])
                for i in range(2)  # [A, B] = 0.2
            ],
            [
                Interaction(["A", "C"], [1.0, 0])
                for i in range(6)  # [A, C] = 0.6
            ],
            [
                Interaction(["B", "A"], [1.0, 0.0])
                for i in range(8)  # [B, A] = 0.8
            ],
            [
                Interaction(["B", "C"], [1.0, 0])
                for i in range(6)  # [B, C] = 0.6
            ],
            [
                Interaction(["C", "A"], [1.0, 0])
                for i in range(4)  # [C, A] = 0.4
            ],
            [
                Interaction(["C", "B"], [1.0, 0])
                for i in range(4)  # [zen, a_v] = 0.6
            ],
        )

    def test_rock_paper_scissor_with_payoff(self):
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

        ratings = laplacian(interactions)
        ratings = np.array([r.mu for r in ratings])

        assert np.allclose(
            ratings - ratings.max(),
            np.zeros_like(ratings),
            atol=1e-7
        )  # desc: B, A, C

    def test_lanctot_et_al_with_wins(self):
        """
            Test the example from [1].

            [1] Lanctot, Marc, et al. "Evaluating Agents using Social Choice
            Theory." arXiv preprint arXiv:2312.03121 (2023).

        """

        ratings = laplacian(
            list(self._lanctot_et_al_interactions()),
            reduction="wins"
        )
        ratings = np.array([r.mu for r in ratings])

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
