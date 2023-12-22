import json
import argparse
from os.path import dirname

from popcore import Interaction
from poprank.functional.elo import elo, EloRate


def load_fixtures(league: str):
    # Load test data
    d = dirname(__file__)
    clubs_file: str = f"{d}/fixtures/{league}.1.clubs.json"
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


def main(
    league: str, k_factor: float
):
    """
        TODO: docs
    """
    players, interactions = load_fixtures(league=league)
    # Assume the initial rating to be 0 for everyone
    elos: "list[float]" = [EloRate(0, 0) for team in players]

    elos = elo(
        players=players, interactions=interactions,
        elos=elos, k_factor=k_factor, wdl=True
    )

    # Rank the players based on their ratings
    elos, players = [list(t) for t in zip(
            *sorted(zip(elos, players), key=lambda x: x[0].mu, reverse=True))]

    col1, col2 = f"Teams ({league})", f"Ratings ({elo.__name__})"
    print(f"{col1:>30} | {col2:>5}")
    print("".ljust(50, "-"))
    for e, p in zip(elos, players):
        print(f"{p:>30} | {e.mu:>5}")
    print("".ljust(50, "-"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="es")
    parser.add_argument("--k-factor", default=20, type=float)

    main(
        **vars(parser.parse_args())
    )
