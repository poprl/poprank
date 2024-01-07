import unittest
from itertools import (
    chain, permutations, product
)
import numpy as np

from poprank.functional.rates import (
    multidim_elo, bipartite_multidim_elo, MultidimEloRate
)
from popcore import Interaction


class TestMultidimEloFunctional(unittest.TestCase):
    def test_rock_paper_scissor(self):
        # NOTE: seed the stochasticity of cyclic vector initalization
        np.random.seed(0)
        k = 1
        players = ["s", "r", "p"]
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

        elos = [MultidimEloRate(0, 1, k=k) for p in players]
        next_elos = multidim_elo(
            players, interactions, elos, k=k, lr1=1, lr2=0.1,
            # opponents=players, opponents_elos=elos,
            iterations=600
        )

        for sample in interactions:
            player, opponnent = sample.players
            player_elo = next_elos[players.index(player)]
            opponent_elo = next_elos[players.index(opponnent)]
            player_outcome, opponent_outcome = sample.outcomes

            self.assertAlmostEqual(
                player_outcome, player_elo.predict(opponent_elo), places=3)
            self.assertAlmostEqual(
                opponent_outcome, opponent_elo.predict(player_elo), places=3)

    def test_go_example_from_balduzzi_et_al(self):
        """
            Test match with the Table mElo2 in [1], Sec 3.2.

            [1] Balduzzi, David, et al. “Re-Evaluating Evaluation.”
            Proceedings of the 32nd International Conference on Neural
            Information Processing Systems, 2018, pp. 3272–83
        """
        np.random.seed(0)

        k = 1
        players = ["alpha_go_v", "alpha_go_p", "zen"]
        empirical = np.array([
            [None, 0.7, 0.4],
            [0.3, None, 1.0],
            [0.6, 0.0, None]
        ])
        mElo2 = np.array([
            [None, 0.72, 0.46],
            [0.28, None, 0.98],
            [0.55, 0.02, None]
        ])
        interac = chain(
            [
                Interaction(["alpha_go_v", "alpha_go_p"], [1.0, 0])
                for i in range(7)  # [a_v, a_p] = 0.7
            ],
            [
                Interaction(["alpha_go_v", "zen"], [1.0, 0])
                for i in range(4)  # [a_v, zen] = 0.4
            ],
            [
                Interaction(["alpha_go_p", "alpha_go_v"], [1.0, 0.0])
                for i in range(3)  # [a_p, a_v] = 0.3
            ],
            [
                Interaction(["alpha_go_p", "zen"], [1.0, 0])
                for i in range(10)
            ],
            [
                Interaction(["zen", "alpha_go_v"], [1.0, 0])
                for i in range(6)  # [zen, a_v] = 0.6
            ],
        )

        elos = [MultidimEloRate(1.0, 1, k=k) for p in players]
        next_elos = multidim_elo(
            players, list(interac), elos, k=k, lr1=0.1, lr2=0.01,
            iterations=15000
        )

        for player1, player2 in permutations(range(3), 2):
            win_probability = next_elos[player1].predict(next_elos[player2])
            self.assertAlmostEqual(
                win_probability,
                mElo2[player1, player2],
                places=1
            )
            self.assertAlmostEqual(
                win_probability,
                empirical[player1, player2],
                places=1
            )

    def test_bipartite_miltidimelo(self):
        """
            Test the agent vs task scenario.
        """
        np.random.seed(0)
        k = 1
        players = ["player1", "player2", "player3"]
        tasks = ["task1", "task2"]
        interactions = [
            Interaction(["player1", "task1"], [1, 0]),
            Interaction(["player2", "task1"], [0, 1]),
            Interaction(["player3", "task1"], [1, 0]),
            Interaction(["player1", "task2"], [1, 0]),
            Interaction(["player2", "task2"], [0, 1]),
            Interaction(["player3", "task2"], [1, 0]),
        ]

        agent_vs_task = np.array([
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0]
        ])

        player_elos = [MultidimEloRate(1.0, 1, k=k) for p in players]
        task_elos = [MultidimEloRate(1.0, 1, k=k) for t in tasks]
        player_elos, task_elos = bipartite_multidim_elo(
            players, interactions, player_elos, tasks, task_elos,
            k=k, lr1=1, lr2=0.1
        )

        for player, task in product(range(3), range(2)):
            win_probability = player_elos[player].predict(task_elos[task])
            self.assertAlmostEqual(
                win_probability,
                agent_vs_task[player, task],
                places=1
            )
