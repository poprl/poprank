import json
from os.path import dirname

from popcore import Interaction


def load_fixtures(league: str = "en"):
    # Load test data
    d = dirname(__file__)
    clubs_file: str = f"{d}/data/{league}.1.clubs.json"
    with open(clubs_file, 'r', encoding='UTF-8') as f:
        # Get a list of all club names
        clubs = json.load(f)
        players = [team["name"] for team in clubs["clubs"]]

    interactions_file: str = f"{d}/fixtures/{league}.1.json"
    with open(interactions_file, 'r', encoding='UTF-8') as f:
        # Get the list of all interactions between clubs
        matches = json.load(f)
        interactions: "list[Interaction]" = []

        for match in matches["matches"]:
            player, opponent = match["team1"], match["team2"]
            outcomes: "list[int]" = match["score"]["ft"]
            interactions.append(
                Interaction(
                    [player, opponent],
                    outcomes
                )
            )

    return players, interactions
