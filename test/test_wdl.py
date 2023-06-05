import unittest
from poprank.functional.wdl import windrawlose, winlose
from popcore import Interaction
from poprank import Rate
import json


def fixtures_test(league):

    # Load test data
    with open(f"poprank/test/fixtures/2019/{league}.1.clubs.json", 'r') as f:

        # Get a list of all club names

        clubs: "dict[str, str | dict[str, str]]" = json.load(f)
        names: "list[str]" = []
        for team in clubs["clubs"]:
            names.append(team["name"])

    with open(f"poprank/test/fixtures/2019/{league}.1.json", 'r') as f:

        # Get the list of all interactions between clubs

        matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
        interactions: "list[Interaction]" = []

        for match in matches["matches"]:
            players: "list[str]" = [match["team1"], match["team2"]]
            outcomes: "list[int]" = match["score"]["ft"]
            interactions.append(Interaction(players, outcomes))

    with open(f"poprank/test/fixtures/2019/{league}.1.final.json", 'r') as f:

        # Get the final ranking of all clubs

        final: "list[dict[str, str | Rate]]" = []
        for team in json.load(f)["clubs"]:
            final.append({"id": team["id"],
                          "rating": Rate(team["rating"], 0)})

    # Assume the initial rating to be 0 for everyone
    ratings: "list[float]" = [0.0 for team in names]

    # Test that the data expected and the data calculated calculated match
    ratings = windrawlose(players=names,
                          interactions=interactions,
                          ratings=ratings,
                          win_value=3,
                          draw_value=1,
                          loss_value=0)

    return final, ratings, names


class TestWDLFunctional(unittest.TestCase):

    def test_windrawlose1(self) -> None:
        final, ratings, names = fixtures_test("en")
        
        for team in final:
            self.assertTrue(team["rating"].mu ==
                            ratings[names.index(team["id"])].mu)
            self.assertTrue(team["rating"].std ==
                            ratings[names.index(team["id"])].std)

    def test_windrawlose2(self) -> None:
        final, ratings, names = fixtures_test("es")
        
        for team in final:
            self.assertTrue(team["rating"].mu ==
                            ratings[names.index(team["id"])].mu)
            self.assertTrue(team["rating"].std ==
                            ratings[names.index(team["id"])].std)

    def test_windrawlose3(self) -> None:
        players = [1, 2, 3]  # Players of the wrong type
        interactions = []
        ratings = [0.0, 0.0, 0.0]

        with self.assertRaises(TypeError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value=1,
                        loss_value=0)

    def test_windrawlose4(self) -> None:
        players = ["a", "b", "c"]
        interactions = []
        ratings = ["0.0", "0.0", "0.0"]  # ratings of the wrong type

        with self.assertRaises(TypeError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value=1,
                        loss_value=0)

    def test_windrawlose5(self) -> None:
        players = ["a", "b", "c"]
        interactions = []
        ratings = [0.0, 0.0, 0.0]

        with self.assertRaises(TypeError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value="3",  # win val of the wrong type
                        draw_value=1,
                        loss_value=0)

    def test_windrawlose6(self) -> None:
        players = ["a", "b", "c"]
        interactions = []
        ratings = [0.0, 0.0, 0.0]

        with self.assertRaises(TypeError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value="1",  # draw val of the wrong type
                        loss_value=0)

    def test_windrawlose7(self) -> None:
        players = ["a", "b", "c"]
        interactions = []
        ratings = [0.0, 0.0, 0.0]

        with self.assertRaises(TypeError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value=1,
                        loss_value="0")  # loss val of the wrong type

    def test_windrawlose8(self) -> None:
        players = ["a", "b", "c", "d", "e"]
        interactions = [Interaction(["a", "b", "c", "d", "e"],
                                    outcomes=[5, 5, 4, 4, 1]),
                        Interaction(["a", "b", "c", "d", "e"],
                                    outcomes=[0, 0, 2, 0, 1])]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertTrue(
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value=1,
                        loss_value=0) == [Rate(1, 0), Rate(1, 0), Rate(3, 0),
                                          Rate(0, 0), Rate(0, 0)])

    def test_winlose1(self) -> None:
        players = ["a", "b", "c", "d", "e"]
        interactions = [Interaction(["a", "b", "c", "d", "e"],
                                    outcomes=[5, 5, 4, 4, 1]),
                        Interaction(["a", "b", "c", "d", "e"],
                                    outcomes=[0, 0, 2, 0, 1])]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertTrue(
            winlose(players=players,
                    interactions=interactions,
                    ratings=ratings,
                    win_value=3,
                    loss_value=0) == [Rate(3, 0), Rate(3, 0), Rate(3, 0),
                                      Rate(0, 0), Rate(0, 0)])
