import unittest
from poprank.functional.elo import elo  # , bayeselo
from popcore import Interaction
from poprank import EloRate


class TestEloFunctional(unittest.TestCase):
    """Testcases for the elo function"""

    def play(self,
             players: "list[str]" = ("a", "b"),
             interactions: "list[Interaction]" =
             [Interaction(["a", "b"], [1, 0])],
             elos: "list[EloRate]" = (EloRate(1000, 0), EloRate(1000, 0)),
             k_factor: float = 20,
             expected_results: "list[float]" = (1010.0, 990.0)) -> None:
        """Assert that the results from elo match the expected values"""
        self.assertListEqual(
            # Rounding for floating point tolerance since calculators round in
            # different places, resulting in vry slightly different end ratings
            list(map(lambda x: EloRate(round(x.mu), 0),
                     elo(players, interactions, elos, k_factor))),
            [EloRate(e, 0) for e in expected_results])

    def test_elo_win(self) -> None:
        """Default single interaction win case"""
        self.play()

    def test_elo_draw(self) -> None:
        """Default single interaction draw case"""
        self.play(interactions=[Interaction(["a", "b"], [0.5, 0.5])],
                  expected_results=[1000.0, 1000.0])

    def test_elo_lose(self) -> None:
        """Default single interaction loss case"""
        self.play(interactions=[Interaction(["a", "b"], [0, 1])],
                  expected_results=[990.0, 1010.0])

    def test_elo_len_mismatch(self) -> None:
        """len mismatch between players and elos"""
        with self.assertRaises(ValueError):
            elo(players=["a", "b", "c", "d", "e"],
                interactions=[],
                elos=[EloRate(1613, 0), EloRate(1609, 0),
                      EloRate(1477, 0), EloRate(1388, 0),
                      EloRate(1586, 0), EloRate(1720, 0)],
                k_factor=32)

    def test_elo_tournament(self) -> None:
        """Tournament involving multiple players"""
        self.play(players=["a", "b", "c", "d", "e", "f"],
                  interactions=[Interaction(["a", "b"], [0, 1]),
                                Interaction(["a", "c"], [0.5, 0.5]),
                                Interaction(["a", "d"], [1, 0]),
                                Interaction(["a", "e"], [1, 0]),
                                Interaction(["a", "f"], [0, 1])],
                  elos=[EloRate(1613, 0), EloRate(1609, 0), EloRate(1477, 0),
                        EloRate(1388, 0), EloRate(1586, 0), EloRate(1720, 0)],
                  k_factor=32,
                  expected_results=[1601, 1625, 1483, 1381, 1571, 1731])

    def test_elo_too_many_players(self) -> None:
        """Throws exception when an interaction
        does not involve exactly 2 players"""
        with self.assertRaises(ValueError):
            elo(players=["a", "b", "c", "d", "e", "f"],
                interactions=[Interaction(["a", "b"], [0, 1]),
                              Interaction(["a", "c"], [.5, .5]),
                              # Too many players
                              Interaction(["a", "d", "f"], [1, 0, 1]),
                              Interaction(["a", "e"], [1, 0]),
                              Interaction(["a", "f"], [0, 1])],
                elos=[EloRate(1613, 0), EloRate(1609, 0),
                      EloRate(1477, 0), EloRate(1388, 0),
                      EloRate(1586, 0), EloRate(1720, 0)],
                k_factor=32)

    def test_elo_unknown_player(self) -> None:
        """Throws an exception if a player appears in the
        interactions but is absent from the players list"""
        with self.assertRaises(ValueError):
            elo(players=["a", "b", "c", "e", "f"],
                interactions=[Interaction(["a", "b"], [0, 1]),
                              Interaction(["a", "c"], [.5, .5]),
                              Interaction(["a", "d"], [1, 0]),
                              Interaction(["a", "e"], [1, 0]),
                              Interaction(["a", "f"], [0, 1])],
                elos=[EloRate(1613, 0), EloRate(1609, 0),
                      EloRate(1477, 0), EloRate(1388, 0),
                      EloRate(1586, 0), EloRate(1720, 0)],
                k_factor=32)
