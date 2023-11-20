import json
from os.path import dirname

from poprank.rates import EloRate
from poprank.functional import bayeselo
from popcore.core import Interaction


def str_to_outcome(s: str):
    if s == "1-0":
        return (1, 0)
    if s == "0-1":
        return (0, 1)
    return (0.5, 0.5)


# Load the dataset
games_filepath: str = f"{dirname(__file__)}/fixtures/shortened_games.json"
with open(games_filepath, "r") as f:
    games = json.load(f)

# Keep a list of players and interactions
players = set()
interactions = []

for x in games:
    # Add players to the list if they aren't in already
    players.add(x[0])
    players.add(x[1])

    # Convert interaction outcome to chess format and save it
    interac = Interaction(players=[x[0], x[1]], outcomes=str_to_outcome(x[2]))
    interactions.append(interac)

# Initialize elos to 0
players = list(players)
elos = [EloRate(mu=0., std=0.) for _ in players]

# Compute ratings
elos = bayeselo(players, interactions, elos)

# Rank the players based on their ratings
elos, players = [list(t) for t in zip(
        *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

# Print the results
for e, p in zip(elos, players):
    print(f"{p:>20}: {e.mu:>5}")
