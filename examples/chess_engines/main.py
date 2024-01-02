import json
from os.path import dirname

from poprank.core import EloRate
from poprank.functional import bayeselo
from popcore.core import Interaction


def str_to_outcome(s: str):
    if s == "1-0":
        return (1, 0)
    if s == "0-1":
        return (0, 1)
    return (0.5, 0.5)


def load_fixtures():
    games_filepath: str = f"{dirname(__file__)}/fixtures/shortened_games.json"
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


def main():
    """
        TODO: Add docs.
    """
    # Load the dataset
    players, interactions = load_fixtures()
    # Initialize elos to 0
    players = list(players)
    elos = [EloRate(mu=0., std=0.) for _ in players]

    # Compute Bayeselo ratings
    elos = bayeselo(players, interactions, elos)

    # Rank the players based on their ratings
    elos, players = [list(t) for t in zip(
            *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

    # Print a cute table with the results
    col1, col2 = "Engines", f"Ratings ({bayeselo.__name__})"
    print(f"{col1:>20} | {col2:>5}")
    print("".ljust(50, "-"))
    for e, p in zip(elos, players):
        print(f"{p:>20} | {e.mu:>5}")
    print("".ljust(50, "-"))


if __name__ == "__main__":
    main()
