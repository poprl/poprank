import json
from os.path import dirname
from typing import List

from popcore import Interaction, Player


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


def load_fixture() -> tuple[List[Player], List[Interaction]]:

    # Import tournament dataset.
    # Originally from https://lmsys.org/blog/2023-05-03-arena/

    path = f"{dirname(__file__)}/clean_battle_20230717.json"
    with open(path, 'r') as f:
        file = json.load(f)

    print(f"Loaded {len(file)} matches")

    # Get a list of players and interactions
    players = set()
    interactions = []

    for m in file:
        # Add the players to the list of contenders if they aren't in already
        player, opponent, outcome = m['model_a'], m['model_b'], m['winner']
        players.update([player, opponent])

        # Turn the outcome to win-loss notation
        outcome = str_to_outcome(outcome)

        # Store the interaction
        interactions.append(
            Interaction(
                [player, opponent],
                outcome
            )
        )
    return players, interactions
