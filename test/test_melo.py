import unittest
from poprank.functional import mElo, mEloAvT, MeloRate
from popcore import Interaction
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
        print(.5, round(new_elos[0].predict(new_elos[0]), 3))
        print(1., round(new_elos[0].predict(new_elos[1]), 3))
        print(0., round(new_elos[0].predict(new_elos[2]), 3))
        print(0., round(new_elos[1].predict(new_elos[0]), 3))
        print(.5, round(new_elos[1].predict(new_elos[1]), 3))
        print(1., round(new_elos[1].predict(new_elos[2]), 3))
        print(1., round(new_elos[2].predict(new_elos[0]), 3))
        print(0., round(new_elos[2].predict(new_elos[1]), 3))
        print(.5, round(new_elos[2].predict(new_elos[2]), 3))

    def test_example_from_learning_to_rank_paper(self):
        k = 1
        players = ["a", "b", "c"]
        interac = []
        interac.extend([Interaction(["a", "b"], [1, 0]) for i in range(7)])
        interac.extend([Interaction(["a", "b"], [0, 1]) for i in range(3)])
        interac.extend([Interaction(["a", "c"], [1, 0]) for i in range(4)])
        interac.extend([Interaction(["a", "c"], [0, 1]) for i in range(6)])
        interac.extend([Interaction(["b", "c"], [1, 0]) for i in range(10)])

        for x in range(10):
            interac.extend(interac)

        shuffle(interac)
        elos = [MeloRate(0, 1, k=k) for p in players]
        new_elos = mElo(players, interac, elos, k=k, lr1=0.001, lr2=0.01)
        print()
        print(.5, round(new_elos[0].predict(new_elos[0]), 3))
        print(0.7, round(new_elos[0].predict(new_elos[1]), 3))
        print(0.4, round(new_elos[0].predict(new_elos[2]), 3))
        print(0.3, round(new_elos[1].predict(new_elos[0]), 3))
        print(.5, round(new_elos[1].predict(new_elos[1]), 3))
        print(1.0, round(new_elos[1].predict(new_elos[2]), 3))
        print(0.6, round(new_elos[2].predict(new_elos[0]), 3))
        print(0.0, round(new_elos[2].predict(new_elos[1]), 3))
        print(.5, round(new_elos[2].predict(new_elos[2]), 3))

    def test_agent_against_task(self):
        k = 1
        players = ["a", "b", "c"]
        tasks = ["d", "e"]
        interac = []

        for i in range(100):    # Needs enough cases to converge
            interac.extend([
                Interaction(["a", "d"], [1, 0]),
                Interaction(["b", "d"], [0, 1]),
                Interaction(["c", "d"], [1, 0]),
                Interaction(["a", "e"], [1, 0]),
                Interaction(["b", "e"], [0, 1]),
                Interaction(["c", "e"], [1, 0]),
            ])

        shuffle(interac)

        player_elos = [MeloRate(0, 1, k=k) for p in players]
        task_elos = [MeloRate(0, 1, k=k) for t in tasks]
        player_elos, task_elos = mEloAvT(
            players, tasks, interac, player_elos, task_elos,
            k=k, lr1=1, lr2=0.1)
        print()
        print(1., round(player_elos[0].predict(task_elos[0]), 3))
        print(0., round(player_elos[1].predict(task_elos[0]), 3))
        print(1., round(player_elos[2].predict(task_elos[0]), 3))
        print(1., round(player_elos[0].predict(task_elos[1]), 3))
        print(0., round(player_elos[1].predict(task_elos[1]), 3))
        print(1., round(player_elos[2].predict(task_elos[1]), 3))
