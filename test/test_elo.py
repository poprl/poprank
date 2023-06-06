import unittest
from poprank.functional.elo import elo, bayeselo
from popcore import Interaction
from poprank import Rate


class TestEloFunctional(unittest.TestCase):
    def play(self,
             players: "list[str]" = ["a", "b"],
             interactions: "list[Interaction]" = [Interaction(["a", "b"], [1, 0])],
             elos: "list[Rate]" = [Rate(1000, 0), Rate(1000, 0)],
             k_factor: float = 20,
             expected_results: "list[float]" = [1010, 990]):

        self.assertListEqual(
            elo(players, interactions, elos, k_factor),
            [Rate(e, 0) for e in expected_results])

    def test_elo_win(self):
        self.play()

    def test_elo_draw(self):
        self.play(interactions=[Interaction(["a", "b"], [0.5, 0.5])],
                  expected_results=[1000, 1000])

    def test_elo_lose(self):
        self.play(interactions=[Interaction(["a", "b"], [0, 1])],
                  expected_results=[990, 1010])

    def test_elo_multiplayer(self):
        pass
