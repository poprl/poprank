import unittest
import json
from poprank.functional.elo import elo  # , bayeselo
from popcore import Interaction
from poprank import EloRate
from os.path import dirname


class TestEloFunctional(unittest.TestCase):
    """Testcases for the elo function"""

    def play(self,
             players: "list[str]",
             interactions: "list[Interaction]",
             elos: "list[EloRate]",
             k_factor: float,
             expected_results: "list[float]",
             wdl: bool = False) -> None:
        """Assert that the results from elo match the expected values"""
        self.assertListEqual(
            # Rounding for floating point tolerance since calculators round in
            # different places, resulting in vry slightly different end ratings
            list(map(lambda x: EloRate(round(x.mu), 0),
                     elo(players, interactions, elos, k_factor, wdl))),
            [EloRate(e, 0) for e in expected_results])

    def test_winning_increases_elo(self) -> None:
        """Default single interaction win case"""
        self.play(
            players=["a", "b"],
            interactions=[
                Interaction(["a", "b"], [1, 0])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_results=[
                1010.0,
                990.0
            ]
        )

    def test_drawing_does_not_change_rating_when_even(self) -> None:
        """Default single interaction draw case"""
        self.play(
            players=("a", "b"),
            interactions=[
                Interaction(["a", "b"], [0.5, 0.5])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_results=[
                1000.0,
                1000.0
            ]
        )

    def test_drawing_changes_rating_when_uneven(self) -> None:
        """Default single interaction draw case"""
        self.play(
            players=("a", "b"),
            interactions=[
                Interaction(["a", "b"], [0.5, 0.5])
            ],
            elos=[
                EloRate(900, 0),
                EloRate(1100, 0)
            ],
            k_factor=20,
            expected_results=[
                905,
                1095
            ]
        )

    def test_losing_decreases_elo(self) -> None:
        """Default single interaction loss case"""
        self.play(
            players=["a", "b"],
            interactions=[
                Interaction(["a", "b"], [0, 1])
            ],
            elos=[
                EloRate(1000, 0),
                EloRate(1000, 0)
            ],
            k_factor=20,
            expected_results=[
                990.0,
                1010.0
            ]
        )

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

    def test_multiple_matches_happening_at_once(self) -> None:
        """Tournament involving multiple players"""
        self.play(
            players=["a", "b", "c", "d", "e", "f"],
            interactions=[
                Interaction(["a", "b"], [0, 1]),
                Interaction(["a", "c"], [0.5, 0.5]),
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
            k_factor=32,
            expected_results=[
                1601,
                1625,
                1483,
                1381,
                1571,
                1731
            ]
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
        self.play(
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
            expected_results=[
                1601,
                1625,
                1483,
                1381,
                1571,
                1731
            ],
            wdl=True
        )

    def fixtures_test(self, league: str):
        """Tests the wdl flag"""
        # Load test data
        d = dirname(__file__)
        clubs_file: str = f"{d}/fixtures/2019/{league}.1.clubs.json"
        with open(clubs_file, 'r', encoding='UTF-8') as f:

            # Get a list of all club names
            names: "list[str]" = \
                [team["name"] for team in json.load(f)["clubs"]]

        interactions_file: str = f"{d}/fixtures/2019/{league}.1.json"
        with open(interactions_file, 'r', encoding='UTF-8') as f:

            # Get the list of all interactions between clubs

            matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
            interactions: "list[Interaction]" = []

            for match in matches["matches"]:
                players: "tuple[str]" = (match["team1"], match["team2"])
                outcomes: "list[int]" = match["score"]["ft"]
                interactions.append(Interaction(players, outcomes))

        # Assume the initial rating to be 0 for everyone
        ratings: "list[float]" = [EloRate(0, 0) for team in names]

        ratings = elo(
            players=names, interactions=interactions,
            elos=ratings, k_factor=32, wdl=True
        )

    def test_computing_elo_from_file_with_wrong_outcome_format(self) -> None:
        """Calculate the elo of the en league from data
        not in the wdl format using the wdl flag"""
        self.fixtures_test("en")

    def test_computing_elo_from_file_with_wrong_outcome_format2(self) -> None:
        """Calculate the elo of the es league from data
        not in the wdl format using the wdl flag"""
        self.fixtures_test("es")
