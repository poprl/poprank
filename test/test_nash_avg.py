import unittest

from popcore import Interaction
from poprank import Rate
from poprank.functional import nash_avg


class TestNashAveraging(unittest.TestCase):

    def test_verify_nashpy_requirement(self):
        try:
            import nashpy  # noqa
        except ImportError:
            self.assertTrue(False)
        self.assertTrue(True)

    def test_verify_zero_sum_game(self):
        self.assertTrue(False)

    def test_verify_rock_paper_scissors(self):
        nash = nash_avg(
            players=["r", "p", "s"],
            interactions=[
                Interaction(
                    players=["r", "p"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["p", "s"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["s", "r"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["r", "r"],
                    outcomes=[0.0, 0.0],
                ),
                Interaction(
                    players=["p", "p"],
                    outcomes=[0.0, 0.0]
                ),
                Interaction(
                    players=["s", "s"],
                    outcomes=[0.0, 0.0]
                )
            ]
        )
        expected_outcome = [
            Rate(1/3),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)

    def test_verify_equilibrium_selection_entropy(self):
        self.assertTrue(False)
