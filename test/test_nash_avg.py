import unittest

from popcore import Interaction
from poprank import Rate
from poprank.functional import nash_avg, nash_avgAvT
from math import floor


class TestNashAveraging(unittest.TestCase):

    def test_verify_nashpy_requirement(self):
        try:
            import nashpy  # noqa
        except ImportError:
            self.assertTrue(False)
        self.assertTrue(True)

    def test_zero_sum_game(self):
        return
        self.assertTrue(False)

    def test_rock_paper_scissors(self):
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

    def test_rock_paper_scissors_linear(self):
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
            ], nash_method="linear"
        )
        expected_outcome = [
            Rate(1/3),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)

    def test_rock_paper_scissors_lemke_howson(self):
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
            ], nash_method="lemke_howson"
        )
        expected_outcome = [
            Rate(1/3),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)

    def test_rock_paper_scissors_lemke_howson_enum(self):
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
            ], nash_method="lemke_howson_enum"
        )
        expected_outcome = [
            Rate(1/3),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)

    def test_rock_paper_scissors_fire_water(self):
        nash = nash_avg(
            players=["r", "p", "s", "w", "f"],
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
                    players=["w", "p"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["w", "s"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["w", "r"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["r", "f"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["p", "f"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["s", "f"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["w", "f"],
                    outcomes=[1.0, -1.0]
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
                ),
                Interaction(
                    players=["w", "w"],
                    outcomes=[0.0, 0.0]
                ),
                Interaction(
                    players=["f", "f"],
                    outcomes=[0.0, 0.0]
                )
            ]
        )
        expected_outcome = [
            Rate(1/9),
            Rate(1/9),
            Rate(1/9),
            Rate(1/3),
            Rate(1/3)
        ]
        self.assertListEqual(nash, expected_outcome)

    def test_rps_n_moves(self):
        n = 9  # n odd, number of moves in this variant of rock-paper-scissors
        players = [str(i) for i in range(n)]
        interactions = []
        interactions.extend(
            [Interaction([str(i), str(i)], [0, 0]) for i in range(n)]
        )

        for i in range(n):
            interactions.extend([
                Interaction(
                    [str(i), players[i-j-1]],
                    [1, -1]
                ) for j in range(floor(n/2))
            ])

        nash = nash_avg(players, interactions)

        expected_outcome = [Rate(1/n) for i in range(n)]
        self.assertListEqual(nash, expected_outcome)

    def test_equilibrium_selection_entropy(self):
        return
        # TODO
        self.assertTrue(False)

    def test_AvT(self):
        players = ["a", "b", "c"]
        tasks = ["d", "e"]
        interac = [
            Interaction(["a", "d"], [1, 0]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [1, 0]),
            Interaction(["a", "e"], [0, 1]),
            Interaction(["b", "e"], [1, 0]),
            Interaction(["c", "e"], [0, 1])]

        # TODO: Vertex doesn't work for some reason?
        player_nash, task_nash = nash_avgAvT(
            players, tasks, interac, nash_method="lemke_howson_enum")
