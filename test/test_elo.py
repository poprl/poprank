import unittest
from poprank.functional.elo import elo, bayeselo
from popcore import Interaction
from poprank import Rate


class TestEloFunctional(unittest.TestCase):
    """Testcases for the elo function"""

    def play(self,
             players: "list[str]" = ("a", "b"),
             interactions: "list[Interaction]" =
             (Interaction(["a", "b"], [1, 0])),
             elos: "list[Rate]" = (Rate(1000, 0), Rate(1000, 0)),
             k_factor: float = 20,
             expected_results: "list[float]" = (1010.0, 990.0)) -> None:
        """Assert that the results from elo match the expected values"""
        self.assertListEqual(
            # Rounding for floating point tolerance since calculators round in
            # different places, resulting in vry slightly different end ratings
            list(map(lambda x: Rate(round(x.mu), 0),
                     elo(players, interactions, elos, k_factor))),
            [Rate(e, 0) for e in expected_results])

    def test_elo_win(self):
        """Default single interaction win case"""
        self.play()

    def test_elo_draw(self):
        """Default single interaction draw case"""
        self.play(interactions=[Interaction(["a", "b"], [0.5, 0.5])],
                  expected_results=[1000.0, 1000.0])

    def test_elo_lose(self):
        """Default single interaction loss case"""
        self.play(interactions=[Interaction(["a", "b"], [0, 1])],
                  expected_results=[990.0, 1010.0])

    def test_elo_tournament(self):
        """Tournament involving multiple players"""
        self.play(players=["a", "b", "c", "d", "e", "f"],
                  interactions=[Interaction(["a", "b"], [0, 1]),
                                Interaction(["a", "c"], [0.5, 0.5]),
                                Interaction(["a", "d"], [1, 0]),
                                Interaction(["a", "e"], [1, 0]),
                                Interaction(["a", "f"], [0, 1])],
                  elos=[Rate(1613, 0), Rate(1609, 0), Rate(1477, 0),
                        Rate(1388, 0), Rate(1586, 0), Rate(1720, 0)],
                  k_factor=32,
                  expected_results=[1601, 1625, 1483, 1381, 1571, 1731])

    def test_elo_too_many_players(self):
        """Throws exception when an interaction does not involve exactly 2 players"""
        with self.assertRaises(ValueError):
            elo(players=["a", "b", "c", "d", "e", "f"],
                interactions=[Interaction(["a", "b"], [0, 1]),
                              Interaction(["a", "c"], [.5, .5]),
                              Interaction(["a", "d", "f"], [1, 0, 1]),  # Too many players
                              Interaction(["a", "e"], [1, 0]),
                              Interaction(["a", "f"], [0, 1])],
                elos=[Rate(1613, 0), Rate(1609, 0),
                      Rate(1477, 0), Rate(1388, 0),
                      Rate(1586, 0), Rate(1720, 0)],
                k_factor=32)

    def test_elo_unknown_player(self):
        """Throws an exception if a player appears in """
        with self.assertRaises(ValueError):
            elo(players=["a", "b", "c", "e", "f"],
                interactions=[Interaction(["a", "b"], [0, 1]),
                              Interaction(["a", "c"], [.5, .5]),
                              Interaction(["a", "d", "f"], [1, 0, 1]),
                              Interaction(["a", "e"], [1, 0]),
                              Interaction(["a", "f"], [0, 1])],
                elos=[Rate(1613, 0), Rate(1609, 0),
                      Rate(1477, 0), Rate(1388, 0),
                      Rate(1586, 0), Rate(1720, 0)],
                k_factor=32)

    """The elo from the external data is modded too heavily to be used,
    so the test was discarded
    def test_elo_against_data(self):
        data_path: str = "poprank/test/fixtures/elo_dataset.json"
        with open(data_path, 'r', encoding='UTF-8') as f:
            dataset: "dict[str, list[str]|dict[str, list[str]|str]]" = \
                json.load(f)

        players: "list[list[str]]" = []
        interactions: "list[Interaction]" = []
        ratings_in: "list[list[float]]" = []
        ratings_out: "list[list[float]]" = []
        for game in dataset['games']:
            if game["game_location"] == "N":
                game["points"] = list(map(int, game["points"]))
                game["elo_in"] = list(map(lambda x: Rate(float(x), 0),
                                          game["elo_in"]))
                game["elo_out"] = list(map(lambda x: Rate(float(x), 0),
                                           game["elo_out"]))
                players.append(game["team_ids"])
                interactions.append(Interaction(game["team_ids"],
                                                game["points"]))
                ratings_in.append(game["elo_in"])
                ratings_out.append(game["elo_out"])

        # assert len(interactions) == -1, f"got {len(interactions)}"

        for index, interac in enumerate(interactions):
            self.assertListEqual(elo(players[index], interac,
                                     ratings_in[index], 20),
                                 ratings_out[index])"""
