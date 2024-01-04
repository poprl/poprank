import unittest
import json
from os.path import dirname
from poprank.functional.rates import windrawlose, winlose
from popcore import Interaction
from poprank import Rate


class TestWDLFunctional(unittest.TestCase):
    """Test cases for the windrawlose and winlose functions"""

    def fixtures_test(self, league: str):
        """Tests windrawlose implementation against known data"""
        # Load test data
        d = dirname(__file__)
        clubs_file: str = f"{d}/fixtures/football.{league}.1.clubs.json"
        with open(clubs_file, 'r', encoding='UTF-8') as f:

            # Get a list of all club names
            names: "list[str]" = \
                [team["name"] for team in json.load(f)["clubs"]]

        interactions_file: str = f"{d}/fixtures/football.{league}.1.json"
        with open(interactions_file, 'r', encoding='UTF-8') as f:

            # Get the list of all interactions between clubs

            matches: "dict[str, str | dict[str, str | dict]]" = json.load(f)
            interactions: "list[Interaction]" = []

            for match in matches["matches"]:
                players: "tuple[str]" = (match["team1"], match["team2"])
                outcomes: "list[int]" = match["score"]["ft"]
                interactions.append(Interaction(players, outcomes))

        final_file: str = f"{d}/fixtures/football.{league}.1.final.json"
        with open(final_file, 'r', encoding='UTF-8') as f:

            # Get the final ranking of all clubs

            final: "list[dict[str, str | Rate]]" = []
            for team in json.load(f)["clubs"]:
                final.append({"id": team["id"],
                              "rating": Rate(team["rating"], 0)})

        # Assume the initial rating to be 0 for everyone
        ratings: "list[float]" = [Rate(0, 0) for team in names]

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

    def exception_tests(self,
                        players: "list[str]" = ["a", "b", "c"],
                        interactions: "list[Interaction]" = [],
                        ratings: "list[float]" = [Rate(0, 0), Rate(0, 0),
                                                  Rate(0, 0)],
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

    def test_windrawlose_against_known_values_en(self) -> None:
        """Test implementation against known values in the fixtures folder (en)
        """
        self.fixtures_test("en")

    def test_windrawlose_against_known_values_es(self) -> None:
        """Test implementation against known values in the fixtures folder (es)
        """
        self.fixtures_test("es")

    def test_players_of_wrong_type_raise_an_error(self) -> None:
        """Players of the wrong type"""
        self.exception_tests(players=[1, 2, 3])

    def test_ratings_of_wrong_type_raise_an_error(self) -> None:
        """Ratings of the wrong type"""
        self.exception_tests(ratings=["0.0", "0.0", "0.0"])

    def test_winvalue_of_wrong_type_raise_an_error(self) -> None:
        """ win val of the wrong type"""
        self.exception_tests(win_value="3")

    def test_drawvalue_of_wrong_type_raise_an_error(self) -> None:
        """draw val of the wrong type"""
        self.exception_tests(draw_value="3")

    def test_lossvalue_of_wrong_type_raise_an_error(self) -> None:
        """loss val of the wrong type"""
        self.exception_tests(loss_value="3")

    def test_interactions_between_n_agents(self) -> None:
        """Test windrawlose in a N agent setting"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[5, 5, 4, 4, 1]
            ),
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[0, 0, 2, 0, 1]
            )
        ]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertListEqual(
            windrawlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings],
                win_value=3, draw_value=1, loss_value=0
            ),
            [
                Rate(1, 0), Rate(1, 0), Rate(3, 0),
                Rate(0, 0), Rate(0, 0)
            ]
        )

    def test_interactions_between_n_agents2(self) -> None:
        """Test windrawlose in a N agent setting, general case"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
                Interaction(
                    players=["a", "c", "d", "e"],
                    outcomes=[5, 5, 4, 1]
                ),
                Interaction(
                    players=["a", "c", "b"],
                    outcomes=[-7, 2.4, -3]
                ),
                Interaction(
                    players=["c", "d"],
                    outcomes=[1237.9815, -3176.5541]
                )
        ]
        ratings = [3.0, 1.0, -.5, 2.5, -.5]

        self.assertListEqual(
            windrawlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings],
                win_value=3, draw_value=1, loss_value=-0.5
            ),
            [
                Rate(3.5, 0), Rate(0.5, 0), Rate(6.5, 0),
                Rate(1.5, 0), Rate(-1, 0)
            ]
        )

    def test_draw(self) -> None:
        """Test windrawlose in a N agent setting with a 5-way draw"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
                Interaction(
                    players=["a", "b", "c", "d", "e"],
                    outcomes=[5, 5, 4, 4, 1]
                ),
                Interaction(
                    players=["a", "b", "c", "d", "e"],
                    outcomes=[-3, -3, -3, -3, -3]
                )
        ]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertListEqual(
            windrawlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings], win_value=3,
                draw_value=1, loss_value=0
            ),
            [
                Rate(2, 0), Rate(2, 0), Rate(1, 0),
                Rate(1, 0), Rate(1, 0)
            ]
        )

    def test_length_mismatch_between_players_and_ratings(self) -> None:
        """Length mismatch between players and ratings"""
        players: "list[str]" = ["a", "b", "c"]
        interactions: "list[Interaction]" = []
        ratings = [0.0, 0.0]

        with self.assertRaises(ValueError):
            windrawlose(players=players,
                        interactions=interactions,
                        ratings=[Rate(rating, 0) for rating in ratings],
                        win_value=3,
                        draw_value=1,
                        loss_value=0)

    def test_winlose_in_n_agents_setting(self) -> None:
        """Test winlose in N agent setting"""
        players: "list[str]" = ["a", "b", "c", "d", "e"]
        interactions: "list[Interaction]" = [
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[5, 5, 4, 4, 1]
            ),
            Interaction(
                players=["a", "b", "c", "d", "e"],
                outcomes=[0, 0, 2, 0, 1]
            )
        ]
        ratings = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.assertListEqual(
            winlose(
                players=players, interactions=interactions,
                ratings=[Rate(rating, 0) for rating in ratings],
                win_value=3, loss_value=0
            ),
            [
                Rate(3, 0), Rate(3, 0), Rate(3, 0),
                Rate(0, 0), Rate(0, 0)
            ]
        )
