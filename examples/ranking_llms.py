import json
from os.path import dirname

from popcore import Interaction
from poprank.rates import EloRate
from poprank.functional import elo


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


# Import tournament dataset.
# Originally from https://lmsys.org/blog/2023-05-03-arena/

path = f"{dirname(__file__)}/fixtures/clean_battle_20230717.json"
with open(path, 'r') as f:
    file = json.load(f)

print(f"Loaded {len(file)} matches")

# Get a list of players and interactions
players = set()
interactions = []

for match in file:
    # Add the players to the list of contenders if they aren't in already
    players.add(match['model_a'])
    players.add(match['model_b'])

    # Turn the outcome to chess notation
    outcome = str_to_outcome(match['winner'])

    # Store the interaction
    interac = Interaction([match['model_a'], match['model_b']], outcome)
    interactions.append(interac)

# Initialize the elos to 0
players = list(players)
elos = [EloRate(0) for x in players]

# Compute the ratings
elos = elo(players, interactions, elos, k_factor=4)

# Rank the players based on their ratings
elos, players = [list(t) for t in zip(
        *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

# Print the results
for e, p in zip(elos, players):
    print(f"{p:>20}: {e.mu:>5}")
