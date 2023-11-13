import json
import os

from popcore import Interaction
from poprank.rates import EloRate
from poprank.functional import bayeselo


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


# Import tournament dataset.
# Originally from https://lmsys.org/blog/2023-05-03-arena/

FILENAME = "clean_battle_20230717.json"
path = os.path.join(os.path.dirname(__file__), FILENAME)
with open(path) as f:
    file = json.load(f)

print(f"Loaded {len(file)} matches")

# Get a list of players and interactions
players = set()
interactions = []

for match in file:
    players.add(match['model_a'])
    players.add(match['model_b'])

    outcome = str_to_outcome(match['winner'])

    interac = Interaction([match['model_a'], match['model_b']], outcome)

    interactions.append(interac)

players = list(players)
elos = [EloRate(0) for x in players]

elos = bayeselo(players, interactions, elos)

# Rank the players based on their ratings

elos, players = [list(t) for t in zip(
        *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

for e, p in zip(elos, players):
    print(f"{p:>20}: {e.mu:>5}")
