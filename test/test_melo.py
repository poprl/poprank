import unittest
from poprank.functional import mElo
from popcore import Interaction
from poprank.rates import MeloRate


class TestEloFunctional(unittest.TestCase):
    def test_play(self):
        k = 2
        players = ["a", "b", "c"]
        interactions = [
            Interaction(["a", "b"], [1, 0]),
            Interaction(["b", "c"], [1, 0]),
            Interaction(["c", "a"], [1, 0])
        ]
        elos = [MeloRate(0, 1, k=k) for p in players]
        new_elos = mElo(players, interactions, elos, iterations=100, k=k)
        print()
        print(.5, round(new_elos[0].expected_outcome(new_elos[0]), 3))
        print(1., round(new_elos[0].expected_outcome(new_elos[1]), 3))
        print(0., round(new_elos[0].expected_outcome(new_elos[2]), 3))
        print(0., round(new_elos[1].expected_outcome(new_elos[0]), 3))
        print(.5, round(new_elos[1].expected_outcome(new_elos[1]), 3))
        print(1., round(new_elos[1].expected_outcome(new_elos[2]), 3))
        print(1., round(new_elos[2].expected_outcome(new_elos[0]), 3))
        print(0., round(new_elos[2].expected_outcome(new_elos[1]), 3))
        print(.5, round(new_elos[2].expected_outcome(new_elos[2]), 3))
