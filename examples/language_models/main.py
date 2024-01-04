import json
from os.path import dirname

from popcore import Interaction
from poprank.core import EloRate
from poprank.functional.rates import elo, bayeselo


def str_to_outcome(s: str):
    """Turn the string describing the outcome into chess notation"""
    if s == "model_a":
        return (1, 0)
    if s == "model_b":
        return (0, 1)
    return (0.5, 0.5)


def load_fixtures():

    # Import tournament dataset.
    # Originally from https://lmsys.org/blog/2023-05-03-arena/

    path = f"{dirname(__file__)}/fixtures/clean_battle_20230717.json"
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


def main():
    # Initialize the elos to 0
    players, interactions = load_fixtures()

    players = list(players)
    elos = [EloRate(0) for x in players]

    # Compute the ratings
    # (Note that you could also give all interactions at once, but then it would
    # only update ratings at the end of all the interactions rather than after each
    # interaction, which does not give the same rating)
    # for match in interactions:
    elos = elo(players, interactions, elos)

    # Rank the players based on their ratings
    elos, players = [list(t) for t in zip(
            *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

    col1, col2 = "Language Models", f"Ratings ({elo.__name__})"
    print(f"{col1:>30} | {col2:>5}")
    print("".ljust(50, "-"))
    for e, p in zip(elos, players):
        print(f"{p:>30} | {e.mu:>5}")
    print("".ljust(50, "-"))


if __name__ == "__main__":
    main()
