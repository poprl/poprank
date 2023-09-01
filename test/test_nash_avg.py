import unittest

from popcore import Interaction
from poprank import Rate
from poprank.functional import nash_avg



class TestNashAveraging(unittest.TestCase):

    def test_verify_nashpy_requirement(self):
        self.assertTrue(False)

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
                )
            ]
        )
        expected_outcome = [
            Rate(1/3),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)
