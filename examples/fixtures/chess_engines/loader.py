import json
from os.path import dirname

from popcore import Interaction


def str_to_outcome(s: str):
    if s == "1-0":
        return 1, 0
    if s == "0-1":
        return 0, 1
    return 0.5, 0.5


def load_fixtures(dataset: str):
    if dataset == "long":
        games_filepath = f"{dirname(__file__)}/computer_chess.500k.json"
    elif dataset == "short":
        games_filepath = f"{dirname(__file__)}/shortened_games.json"
    else:
        raise ValueError()

    with open(games_filepath, "r") as f:
        games = json.load(f)

    # Keep a list of players and interactions
    players = set()
    interactions = []
    for game in games:
        # Add players to the list if they aren't in already
        player, opponent, outcome = game

        players.update([player, opponent])
        # Convert chess string format to
        # win-loss outcome format and save it
        interactions.append(
            Interaction(
                players=[player, opponent],
                outcomes=str_to_outcome(outcome)
            )
        )

    return players, interactions
