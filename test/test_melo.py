import unittest
from poprank.functional import mElo
from popcore import Interaction
from poprank.rates import MeloRate
from random import shuffle


class TestEloFunctional(unittest.TestCase):
    def test_rock_paper_scissor(self):
        k = 1
        players = ["a", "b", "c"]
        interactions = []

        for i in range(100):    # Needs enough cases to converge
            interactions.extend([
                Interaction(["a", "b"], [1, 0]),
                Interaction(["b", "c"], [1, 0]),
                Interaction(["c", "a"], [1, 0])
            ])

        elos = [MeloRate(0, 1, k=k) for p in players]
        new_elos = mElo(players, interactions, elos, k=k, lr1=1, lr2=0.1)
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

    def test_from_learning_to_rank_paper(self):
        k = 1
        players = ["a", "b", "c"]
        interactions = []
        interactions.extend([Interaction(["a", "b"], [1, 0]) for i in range(7)])
        interactions.extend([Interaction(["a", "b"], [0, 1]) for i in range(3)])
        interactions.extend([Interaction(["a", "c"], [1, 0]) for i in range(4)])
        interactions.extend([Interaction(["a", "c"], [0, 1]) for i in range(6)])
        interactions.extend([Interaction(["b", "c"], [1, 0]) for i in range(10)])

        for x in range(10):
            interactions.extend(interactions)

        shuffle(interactions)
        elos = [MeloRate(0, 1, k=k) for p in players]
        new_elos = mElo(players, interactions, elos, k=k, lr1=0.001, lr2=0.01)
        print()
        print(.5, round(new_elos[0].expected_outcome(new_elos[0]), 3))
        print(0.7, round(new_elos[0].expected_outcome(new_elos[1]), 3))
        print(0.4, round(new_elos[0].expected_outcome(new_elos[2]), 3))
        print(0.3, round(new_elos[1].expected_outcome(new_elos[0]), 3))
        print(.5, round(new_elos[1].expected_outcome(new_elos[1]), 3))
        print(1.0, round(new_elos[1].expected_outcome(new_elos[2]), 3))
        print(0.6, round(new_elos[2].expected_outcome(new_elos[0]), 3))
        print(0.0, round(new_elos[2].expected_outcome(new_elos[1]), 3))
        print(.5, round(new_elos[2].expected_outcome(new_elos[2]), 3))
