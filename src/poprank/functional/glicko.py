from copy import deepcopy
from math import sqrt, log, exp
from popcore import Interaction
from poprank import GlickoRate


def glicko(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[GlickoRate]", c: float = 34.6,
    RD_unrated: float = 350., base: float = 10., spread: float = 400.
) -> "list[GlickoRate]":

    new_ratings: "list[GlickoRate]" = []

    # Update rating deviations
    for rating in ratings:
        new_ratings.append(
            GlickoRate(rating.mu,
                       min(sqrt(rating.std**2 +
                                rating.time_since_last_competition*c**2),
                           RD_unrated)))
        new_ratings[-1].base = rating.base
        new_ratings[-1].spread = rating.spread
        new_ratings[-1].volatility = rating.volatility
        # Implicitly reset rating time since last competition to 0

    q: float = log(base)/spread
    sum_games: "list[float]" = [0. for p in players]
    d_squared: "list[float]" = [0. for p in players]

    for interaction in interactions:
        id_player0 = players.index(interaction.players[0])
        id_player1 = players.index(interaction.players[1])

        if interaction.outcomes[0] > interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (1, 0)
        elif interaction.outcomes[0] < interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (0, 1)
        else:
            match_outcome: "tuple[float]" = (.5, .5)

        g0 = GlickoRate.g(new_ratings[id_player1].std, q)
        g1 = GlickoRate.g(new_ratings[id_player0].std, q)

        expected_outcome0: float = \
            new_ratings[id_player0].glicko1_expected_outcome(
                new_ratings[id_player1])
        expected_outcome1: float = \
            new_ratings[id_player1].glicko1_expected_outcome(
                new_ratings[id_player0])

        sum_games[id_player0] += g0 * (match_outcome[0] - expected_outcome0)
        sum_games[id_player1] += g1 * (match_outcome[1] - expected_outcome1)

        d_squared[id_player0] += (g0**2 * expected_outcome0 *
                                  (1 - expected_outcome0))
        d_squared[id_player1] += (g1**2 * expected_outcome1 *
                                  (1 - expected_outcome1))

    for i, d in enumerate(d_squared):
        d_squared[i] = 1 / (q**2 * d)

    for i, rating in enumerate(players):
        new_rating = new_ratings[i].mu
        tmp = (1/new_ratings[i].std**2) + 1 / d_squared[i]
        new_rating += q/tmp * sum_games[i]
        new_rating_deviation = sqrt(1/tmp)
        new_ratings[i] = new_ratings[i]._replace(mu=new_rating,
                                                 std=new_rating_deviation)

    return new_ratings


def f(x, volatility, delta, std, v, tau):
    a = (1/2) * (exp(x)*(delta**2 - std**2 - v - exp(x)) /
                 (std**2 + v + exp(x))**2)
    b = (x - log(volatility**2))/tau**2
    return a - b


def glicko2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[GlickoRate]", c: float = 34.6,
    RD_unrated: float = 350., base: float = 10., spread: float = 400.,
    tau: float = .5, epsilon: float = 1e-6, min_delta: float = 1e-4,
    average: float = 1500, conversion_var: float = 173.7178
) -> "list[GlickoRate]":

    new_ratings = [deepcopy(r) for r in ratings]
    new_ratings = [r._replace(mu=(r.mu - average)/conversion_var,
                              std=r.std/conversion_var) for r in new_ratings]

    variance: float = [0. for p in players]
    delta: float = [0. for p in players]
    sum_match: float = [0. for p in players]

    for interaction in interactions:
        id_player0 = players.index(interaction.players[0])
        id_player1 = players.index(interaction.players[1])

        if interaction.outcomes[0] > interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (1, 0)
        elif interaction.outcomes[0] < interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (0, 1)
        else:
            match_outcome: "tuple[float]" = (.5, .5)

        g0: float = GlickoRate.g(new_ratings[id_player1].std, 1)
        print(f"g(Thetaj): {g0}")
        g1: float = GlickoRate.g(new_ratings[id_player0].std, 1)

        expected_outcome0: float = \
            new_ratings[id_player0].glicko2_expected_outcome(
                new_ratings[id_player1])
        print(f"E(mu, muj, thetaj): {expected_outcome0}")
        expected_outcome1: float = \
            new_ratings[id_player1].glicko2_expected_outcome(
                new_ratings[id_player0])

        variance[id_player0] += \
            (g0**2) * expected_outcome0 * (1 - expected_outcome0)
        variance[id_player1] += \
            (g1**2) * expected_outcome1 * (1 - expected_outcome1)

        sum_match[id_player0] += g0 * (match_outcome[0] - expected_outcome0)
        sum_match[id_player1] += g1 * (match_outcome[1] - expected_outcome1)

    variance = [1/x for x in variance]
    delta = [a*b for a, b in zip(variance, sum_match)]

    for i, rating in enumerate(new_ratings):

        # Set initial values for iterative algorithm
        alpha = log(rating.volatility ** 2)

        if delta[i]**2 > RD_unrated**2 + variance[i]:
            b = log(delta[i]**2 - RD_unrated**2 - variance)
        else:
            k = 1
            while f(alpha - k * sqrt(tau ** 2), rating.volatility,
                    delta[i], rating.std, variance[i], tau) < 0:
                k += 1
            b = alpha - k * sqrt(tau**2)

        fa = f(alpha, rating.volatility, delta[i],
               rating.std, variance[i], tau)
        fb = f(b, rating.volatility, delta[i],
               rating.std, variance[i], tau)

        # Iterate

        while abs(b - alpha) > epsilon and abs(b - alpha) > min_delta:
            c = alpha + (alpha - b) * fa / (fb - fa)
            fc = f(c, rating.volatility, delta[i],
                   rating.std, variance[i], tau)

            if fc * fb < 0:
                alpha, fa = b, fb
            else:
                fa /= 2

            b, fb = c, fc

        new_volatility = exp(alpha/2)
        new_std = 1 / sqrt(1/(rating.std ** 2 + new_volatility**2) +
                           1/variance[i])
        new_mu = rating.mu + new_std**2 * sum_match[i]

        new_ratings[i].volatility = new_volatility
        new_ratings[i] = new_ratings[i]._replace(mu=new_mu * conversion_var +
                                                 average, std=new_std *
                                                 conversion_var)

    return new_ratings
