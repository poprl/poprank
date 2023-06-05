import unittest
from poprank.functional.wdl import windrawlose
from popcore import Interaction
from poprank import Rate
import json


class TestWDLFunctional(unittest.TestCase):

    def test_windrawlose1(self) -> None:

        # Load test data
        with open("poprank/test/fixtures/2019/en.1.clubs.json", 'r') as f:

            # Get a list of all club names

            en_clubs: "dict[str, str | dict[str, str]]" = json.load(f)
            en_names: "list[str]" = []
            for team in en_clubs["clubs"]:
                en_names.append(team["name"])

        with open("poprank/test/fixtures/2019/en.1.json", 'r') as f:

            # Get the list of all interactions between clubs

            en_matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
            en_interactions: "list[Interaction]" = []

            for match in en_matches["matches"]:
                players: "list[str]" = [match["team1"], match["team2"]]
                outcomes: "list[int]" = match["score"]["ft"]
                en_interactions.append(Interaction(players, outcomes))

        with open("poprank/test/fixtures/2019/en.1.final.json", 'r') as f:

            # Get the final ranking of all clubs

            en_final: "list[dict[str, str | Rate]]" = []
            for team in json.load(f)["clubs"]:
                en_final.append({"id": team["id"],
                                 "rating": Rate(team["rating"], 0)})

        # Assume the initial rating to be 0 for everyone
        en_ratings: "list[float]" = [0.0 for team in en_names]

        # Test that the data expected and the data calculated calculated match
        en_ratings = windrawlose(players=en_names,
                                 interactions=en_interactions,
                                 ratings=en_ratings,
                                 win_value=3,
                                 draw_value=1,
                                 loss_value=0)

        for team in en_final["clubs"]:
            self.assertTrue(team["rating"].mu ==
                            en_ratings[en_clubs.index(team["id"])].mu)
            self.assertTrue(team["rating"].std ==
                            en_ratings[en_clubs.index(team["id"])].std)

    def test_windrawlose2(self) -> None:

        # Load test data
        with open("poprank/test/fixtures/2019/es.1.clubs.json", 'r') as f:

            # Get a list of all club names

            es_clubs: "dict[str, str | dict[str, str]]" = json.load(f)
            es_names: "list[str]" = []
            for team in es_clubs["clubs"]:
                es_names.append(team["name"])

        with open("poprank/test/fixtures/2019/es.1.json", 'r') as f:

            # Get the list of all interactions between clubs

            es_matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
            es_interactions: "list[Interaction]" = []

            for match in es_matches["matches"]:
                players: "list[str]" = [match["team1"], match["team2"]]
                outcomes: "list[int]" = match["score"]["ft"]
                es_interactions.append(Interaction(players, outcomes))

        with open("poprank/test/fixtures/2019/es.1.final.json", 'r') as f:

            # Get the final ranking of all clubs

            es_final: "list[dict[str, str | Rate]]" = []
            for team in json.load(f)["clubs"]:
                es_final.append({"id": team["id"],
                                 "rating": Rate(team["rating"], 0)})

        # Assume the initial rating to be 0 for everyone
        es_ratings: "list[float]" = [0.0 for team in es_names]

        # Test that the data expected and calculated match
        es_ratings = windrawlose(players=es_names,
                                 interactions=es_interactions,
                                 ratings=es_ratings,
                                 win_value=3,
                                 draw_value=1,
                                 loss_value=0)

        for team in es_final["clubs"]:
            self.assertTrue(team["rating"].mu ==
                            es_ratings[es_clubs.index(team["id"])].mu)
            self.assertTrue(team["rating"].std ==
                            es_ratings[es_clubs.index(team["id"])].std)

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
