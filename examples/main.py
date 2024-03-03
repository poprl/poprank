import argparse
import functools
import time

from poprank import Rate
from poprank.functional.rates import (
    elo, bayeselo, glicko, glicko2, multidim_elo, nash_avg,
    rectified_nash_avg, windrawlose, trueskill, winlose,
)
from poprank.functional.rates.experimental import (
    laplacian
)

from fixtures.chess_engines.loader import load_fixtures as chess_engine_loader
from fixtures.football_leagues.loader import load_fixtures as football_loader
from fixtures.language_models.loader import load_fixture as llm_loader


RATINGS = {
    "elo": elo,
    "stream_elo": functools.partial(elo, reduce="stream"),
    "bayeselo": bayeselo,
    "glicko": glicko,
    "glicko2": glicko2,
    "trueskill": trueskill,
    "multidim_elo": functools.partial(multidim_elo, iterations=5),
    "nash_avg": nash_avg,
    "rectified_nash_avg": rectified_nash_avg,
    "windrawlose": functools.partial(
        windrawlose, win_value=3.0, draw_value=1.0, loss_value=0.0
    ),
    "winlose": functools.partial(
        winlose, win_value=1.0, loss_value=-1.0
    ),
    "laplacian": laplacian
}

FIXTURES = {
    "language-models": llm_loader,
    "football-en": functools.partial(
        football_loader, league="en"),
    "football-es": functools.partial(
        football_loader, league="en"),
    "chess-engines-short": functools.partial(
        chess_engine_loader, dataset="short"),
    "chess-engines-long": functools.partial(
        chess_engine_loader, dataset="long"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=list(FIXTURES.keys()),
        default="language-models")
    parser.add_argument(
        "--method",
        choices=["all"] + list(RATINGS.keys()),
        default="all"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the elos to 0
    fixture_loader = FIXTURES[args.source]
    players, interactions = fixture_loader()
    players = list(players)

    methods = RATINGS.keys() if args.method == "all" else [args.method]

    for method in methods:
        rates = [Rate(0.0) for x in players]
        rating_method = RATINGS[method]
        t1 = time.time()
        rates = rating_method(players, interactions, rates)
        t2 = time.time()
        rates, players = [list(t) for t in zip(
                *sorted(zip(rates, players), key=lambda x: x[0].mu, reverse=True))]

        col1, col2 = "model", f"{method}"
        print(f"{col1:>30} | {col2:>5}")
        print("".ljust(50, "-"))
        for e, p in zip(rates, players):
            print(f"{p:>30} | {e.mu:>5}")
        print("".ljust(50, "-"))
        print(f"rate {method}, time: {t2 - t1} seconds")


if __name__ == "__main__":
    main()
