import unittest
import json
from poprank.functional.elo import elo, bayeselo
from popcore import Interaction
from poprank import Rate


class TestEloFunctional(unittest.TestCase):
    def play(self,
             players: "list[str]" = ["a", "b"],
             interactions: "list[Interaction]" =
             [Interaction(["a", "b"], [1, 0])],
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

    def test_elo_tournament(self):
        self.play(players=["a", "b", "c", "d", "e", "f"],
                  interactions=[Interaction(["a", "b"], [0, 1]),
                                Interaction(["a", "c"], [0.5, 0.5]),
                                Interaction(["a", "d"], [1, 0]),
                                Interaction(["a", "e"], [1, 0]),
                                Interaction(["a", "f"], [0, 1])],
                  elos=[Rate(1613, 0), Rate(1609, 0), Rate(1477, 0),
                        Rate(1388, 0), Rate(1586, 0), Rate(1720, 0)],
                  k_factor=32,
                  expected_results=[Rate(1601, 0), Rate(1609+16.32, 0),
                                    Rate(1477+5.76, 0), Rate(1388-6.72, 0),
                                    Rate(1586+14.72, 0), Rate(1720-11.2, 0)])

# test test

    """Test implies taking margin of error into account so it was discarded
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
