import unittest

from poprank.functional.elo import elo
from popcore import Interaction
from poprank import EloRate


class TestEloFunctional(unittest.TestCase):
    """Testcases for the elo function"""

    def _assert_elo_from_interactions(
        self,
        players: "list[str]", interactions: "list[Interaction]",
        elos: "list[EloRate]", k_factor: float,
        expected_elos: "list[float]", wdl: bool = False
    ) -> None:
        """
            Asserts that the results from elo match the expected values.
            Rounding for floating point tolerance since calculators round in
            different places, resulting in vry slightly different end ratings
        """
        elos = elo(players, interactions, elos, k_factor, wdl)
        elos = map(lambda x: EloRate(round(x.mu), 0), elos)
        elos = list(elos)

        self.assertListEqual(
            elos,
            [EloRate(e, 0) for e in expected_elos]
        )

    def test_rock_paper_scissors_should_produce_equal_ratings(self) -> None:
        self._assert_elo_from_interactions(
            players=["R", "P", "S"],
            interactions=[
                Interaction(["R", "P"], [-1, 1]),
                Interaction(["R", "S"], [1, -1]),
                Interaction(["R", "R"], [0, 0]),
                Interaction(["P", "S"], [-1, 1]),
                Interaction(["P", "R"], [1, -1]),
                Interaction(["P", "P"], [0, 0]),
                Interaction(["S", "P"], [1, -1]),
                Interaction(["S", "R"], [-1, 1]),
                Interaction(["S", "S"], [0, 0])
            ],
            elos=[
                EloRate(1.0) for _ in ["R", "P", "S"]
            ],
            k_factor=1.0,
            expected_elos=[1.0, 1.0, 1.0],
            wdl=True
        )

    def test_wikipedia_computation_example(self):
        """
            See: https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details # noqa
        """

        elos = elo(
            players=["A", "B", "C", "D", "E", "F"],
            interactions=[
                Interaction(["A", "B"], [0.0, 1.0]),
                Interaction(["A", "C"], [0.5, 0.5]),
                Interaction(["A", "D"], [1.0, 0.0]),
                Interaction(["A", "E"], [1.0, 0.0]),
                Interaction(["A", "F"], [0.0, 1.0])
            ],
            elos=[
                EloRate(1613),
                EloRate(1609),
                EloRate(1477),
                EloRate(1388),
                EloRate(1586),
                EloRate(1720)
            ],
            k_factor=32
        )
        self.assertEqual(round(elos[0].mu), 1601)

    def test_winning_increases_elo(self) -> None:
        """Default single interaction win case"""
        self._assert_elo_from_interactions(
            players=["a", "b"],
            interactions=[
                Interaction(["a", "b"], [1, 0])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_elos=[
                1010.0,
                990.0
            ]
        )

    def test_drawing_does_not_change_rating_when_even(self) -> None:
        """Default single interaction draw case"""
        self._assert_elo_from_interactions(
            players=("a", "b"),
            interactions=[
                Interaction(["a", "b"], [0.5, 0.5])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_elos=[
                1000.0,
                1000.0
            ]
        )

    def test_draw_decreases_rating_when_uneven(self) -> None:
        """Default single interaction draw case"""
        self._assert_elo_from_interactions(
            players=("a", "b"),
            interactions=[
                Interaction(["a", "b"], [0.5, 0.5])
            ],
            elos=[
                EloRate(900, 0),
                EloRate(1100, 0)
            ],
            k_factor=20,
            expected_elos=[
                905,
                1095
            ]
        )

    def test_losing_decreases_rating_when_even(self) -> None:
        """Default single interaction loss case"""
        self._assert_elo_from_interactions(
            players=["a", "b"],
            interactions=[
                Interaction(["a", "b"], [0, 1])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_elos=[
                990.0,
                1010.0
            ]
        )


class TestEloInterface(unittest.TestCase):

    def test_player_and_elos_length_mismatch_raises_error(self) -> None:
        """len mismatch between players and elos"""
        with self.assertRaises(ValueError):
            elo(
                players=["a", "b", "c", "d", "e"],
                interactions=[],
                elos=[
                    EloRate(1613, 0),
                    EloRate(1609, 0),
                    EloRate(1477, 0),
                    EloRate(1388, 0),
                    EloRate(1586, 0),
                    EloRate(1720, 0)
                ],
                k_factor=32
            )

    def test_interaction_with_more_than_2_players_raises_error(self) -> None:
        """Throws exception when an interaction
        does not involve exactly 2 players"""
        with self.assertRaises(ValueError):
            elo(
                players=["a", "b", "c", "d", "e", "f"],
                interactions=[
                    Interaction(["a", "b"], [0, 1]),
                    Interaction(["a", "c"], [.5, .5]),
                    # Too many players
                    Interaction(["a", "d", "f"], [1, 0, 1]),
                    Interaction(["a", "e"], [1, 0]),
                    Interaction(["a", "f"], [0, 1])
                ],
                elos=[
                    EloRate(1613, 0),
                    EloRate(1609, 0),
                    EloRate(1477, 0),
                    EloRate(1388, 0),
                    EloRate(1586, 0),
                    EloRate(1720, 0),
                ],
                k_factor=32
            )

    def test_an_unknown_player_in_interactions_raises_an_error(self) -> None:
        """Throws an exception if a player appears in the
        interactions but is absent from the players list"""
        with self.assertRaises(ValueError):
            elo(
                players=["a", "b", "c", "e", "f"],
                interactions=[
                    Interaction(["a", "b"], [0, 1]),
                    Interaction(["a", "c"], [.5, .5]),
                    Interaction(["a", "d"], [1, 0]),
                    Interaction(["a", "e"], [1, 0]),
                    Interaction(["a", "f"], [0, 1])
                ],
                elos=[
                    EloRate(1613, 0),
                    EloRate(1609, 0),
                    EloRate(1477, 0),
                    EloRate(1388, 0),
                    EloRate(1586, 0),
                    EloRate(1720, 0)
                ],
                k_factor=32
            )

    def test_converting_outcomes_to_windrawlose_format_works(self):
        elos = elo(
            players=["a", "b", "c", "d", "e", "f"],
            interactions=[
                Interaction(["a", "b"], [8, 10]),
                Interaction(["a", "c"], [0, 0]),
                Interaction(["a", "d"], [9, -3.8]),
                Interaction(["a", "e"], [9, 2]),
                Interaction(["a", "f"], [0, 1])
            ],
            elos=[
                EloRate(1613, 0),
                EloRate(1609, 0),
                EloRate(1477, 0),
                EloRate(1388, 0),
                EloRate(1586, 0),
                EloRate(1720, 0)
            ],
            k_factor=32,
            wdl=True
        )
        elos = map(lambda x: EloRate(round(x.mu), 0), elos)
        elos = list(elos)

        expected_elos = [
            1601,
            1625,
            1483,
            1381,
            1571,
            1731
        ]

        self.assertListEqual(
            elos,
            [EloRate(e, 0) for e in expected_elos]
        )
