import unittest
from poprank.functional.wdl import windrawlose, winlose
from popcore import Interaction
from poprank import Rate
import json


def fixtures_test(self: object, league: str):
    """Tests windrawlose implementation against known data"""
    # Load test data
    clubs_file: str = f"poprank/test/fixtures/2019/{league}.1.clubs.json"
    with open(clubs_file, 'r', encoding='UTF-8') as f:

        # Get a list of all club names
        names: "list[str]" = [team["name"] for team in json.load(f)["clubs"]]

    interactions_file: str = f"poprank/test/fixtures/2019/{league}.1.json"
    with open(interactions_file, 'r', encoding='UTF-8') as f:

        # Get the list of all interactions between clubs

        matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
        interactions: "list[Interaction]" = []

        for match in matches["matches"]:
            players: "tuple[str]" = (match["team1"], match["team2"])
            outcomes: "list[int]" = match["score"]["ft"]
            interactions.append(Interaction(players, outcomes))

    final_file: str = f"poprank/test/fixtures/2019/{league}.1.final.json"
    with open(final_file, 'r', encoding='UTF-8') as f:

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

    for team in final:
        self.assertTrue(team["rating"].mu ==
                        ratings[names.index(team["id"])].mu)
        self.assertTrue(team["rating"].std ==
                        ratings[names.index(team["id"])].std)


def exception_tests(self: object,
                    players: "list[str]" = ["a", "b", "c"],
                    interactions: "list[Interaction]" = [],
                    ratings: "list[float]" = [0.0, 0.0, 0.0],
                    win_value: float = 3,
                    draw_value: float = 1,
                    loss_value: float = 0):
    """Tests typechecking for the windrawlose arguments"""
    with self.assertRaises(AssertionError):
        windrawlose(players=players,
                    interactions=interactions,
                    ratings=ratings,
                    win_value=win_value,
                    draw_value=draw_value,
                    loss_value=loss_value)


class TestWDLFunctional(unittest.TestCase):

    def test_windrawlose_against_known_values_en(self) -> None:
        """Test implementation against known values in the fixtures folder (en)
        """
        fixtures_test(self, "en")

    def test_windrawlose_against_known_values_es(self) -> None:
        """Test implementation against known values in the fixtures folder (es)
        """
        fixtures_test(self, "es")

    def test_windrawlose_player_type(self) -> None:
        """Players of the wrong type"""
        exception_tests(self, players=[1, 2, 3])

    def test_windrawlose_rating_type(self) -> None:
        """Ratings of the wrong type"""
        exception_tests(self, ratings=["0.0", "0.0", "0.0"])

    def test_windrawlose_winvalue_type(self) -> None:
        """ win val of the wrong type"""
        exception_tests(self, win_value="3")

    def test_windrawlose_drawvalue_type(self) -> None:
        """draw val of the wrong type"""
        exception_tests(self, draw_value="3")

    def test_windrawlose_lossvalue_type(self) -> None:
        """loss val of the wrong type"""
        exception_tests(self, loss_value="3")

    def test_windrawlose_n_agents(self) -> None:
        """Test windrawlose in a N agent setting"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = \
            [Interaction(["a", "b", "c", "d", "e"],
             outcomes=[5, 5, 4, 4, 1]),
             Interaction(["a", "b", "c", "d", "e"],
             outcomes=[0, 0, 2, 0, 1])]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertListEqual(
            windrawlose(players=players, interactions=interactions,
                        ratings=ratings, win_value=3, draw_value=1,
                        loss_value=0),
            [Rate(1, 0), Rate(1, 0), Rate(3, 0), Rate(0, 0), Rate(0, 0)])

    def test_windrawlose_player_rating_mismatch(self) -> None:
        """Length mismatch between players and ratings"""
        players: "list[str]" = ["a", "b", "c"]
        interactions: "list[Interaction]" = []
        ratings = [0.0, 0.0]

        with self.assertRaises(ValueError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=ratings,
                        win_value=3,
                        draw_value=1,
                        loss_value=0)

    def test_winlose_n_agent(self) -> None:
        """Test winlose in N agent setting"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = \
            [Interaction(["a", "b", "c", "d", "e"],
             outcomes=[5, 5, 4, 4, 1]),
             Interaction(["a", "b", "c", "d", "e"],
             outcomes=[0, 0, 2, 0, 1])]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertListEqual(
            winlose(players=players, interactions=interactions,
                    ratings=ratings, win_value=3, loss_value=0),
            [Rate(3, 0), Rate(3, 0), Rate(3, 0), Rate(0, 0), Rate(0, 0)])
