from copy import deepcopy
from math import sqrt, log, exp
from popcore import Interaction
from poprank import Glicko1Rate, Glicko2Rate


def glicko(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Glicko1Rate]", uncertainty_increase: float = 34.6,
    rating_deviation_unrated: float = 350., base: float = 10.,
    spread: float = 400.
) -> "list[Glicko1Rate]":
    """Rates players by calculating their new glicko after a set of
    interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5). Interactions
    in different formats are converted automatically if possible.

    See also: :meth:`poprank.functional.glicko.glicko2`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
            get a rating from. Every interaction should be between exactly 2
            players.
        ratings (list[Glicko1Rate]): the initial ratings of the players.
        uncertainty_increase (float, optional): constant governing the
            increase in uncerntainty between rating periods. Defaults to 34.6.
        rating_deviation_unrated (float, optional): The rating deviation of
            unrated players. Defaults to 350.0.
        base (float, optional): Value in the logarithm for the constant q.
            Defaults to 10.0.
        spread (float, optional): Denominator in the constant q.
            Defaults to 400.0.

    Returns:
        list[Glicko1Rate]: the updated ratings of all players
    """

    new_ratings: "list[Glicko1Rate]" = []

    # Update rating deviations
    for rating in ratings:
        new_ratings.append(
            Glicko1Rate(rating.mu,
                        min(sqrt(rating.std**2 +
                                 rating.time_since_last_competition *
                                 uncertainty_increase**2),
                            rating_deviation_unrated)))
        new_ratings[-1].base = rating.base
        new_ratings[-1].spread = rating.spread
        # Implicitly reset rating time since last competition to 0

    # Calculate the variables needed to update ratings and rating deviations

    q: float = log(base)/spread
    total_games_results: "list[float]" = [0. for p in players]
    d_squared: "list[float]" = [0. for p in players]

    for interaction in interactions:
        id_player: int = players.index(interaction.players[0])
        id_opponent: int = players.index(interaction.players[1])

        # Turn interactions outcomes into the (1, 0), (0, 1), (.5, .5) format

        if interaction.outcomes[0] > interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (1, 0)
        elif interaction.outcomes[0] < interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (0, 1)
        else:
            match_outcome: "tuple[float]" = (.5, .5)

        # Calculate the rating adjustment for both players based on expected
        # outcome, actual outcome and rating deviation

        # Factor reducing the impact of games based on opponent rating
        # deviation (Higher RD means lower impact)
        reduce_impact_player: float = \
            Glicko1Rate.reduce_impact(new_ratings[id_opponent].std, q)
        reduce_impact_opponent: float = \
            Glicko1Rate.reduce_impact(new_ratings[id_player].std, q)

        expected_outcome_player: float = \
            new_ratings[id_player].expected_outcome(
                new_ratings[id_opponent])
        expected_outcome_opponent: float = \
            new_ratings[id_opponent].expected_outcome(
                new_ratings[id_player])

        total_games_results[id_player] += reduce_impact_player * \
            (match_outcome[0] - expected_outcome_player)
        total_games_results[id_opponent] += reduce_impact_opponent * \
            (match_outcome[1] - expected_outcome_opponent)

        d_squared[id_player] += (reduce_impact_player**2 *
                                 expected_outcome_player *
                                 (1 - expected_outcome_player))
        d_squared[id_opponent] += (reduce_impact_opponent**2 *
                                   expected_outcome_opponent *
                                   (1 - expected_outcome_opponent))

    # Set d_squared to None if the player did not have any interaction
    for i, d in enumerate(d_squared):
        d_squared[i] = 1 / (q**2 * d) if d != 0 else None

    # Update the ratings and rating deviations

    for i, rating in enumerate(players):
        # Only update if the player had interactions
        if d_squared[i] is not None:
            new_rating: float = new_ratings[i].mu
            denominator: float = (1/new_ratings[i].std**2) + 1 / d_squared[i]
            new_rating += q/denominator * total_games_results[i]
            new_rating_deviation: float = sqrt(1/denominator)
            new_ratings[i] = new_ratings[i]._replace(mu=new_rating,
                                                     std=new_rating_deviation)

    return new_ratings


def glicko2(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Glicko2Rate]", rating_deviation_unrated: float = 350.,
    volatility_constraint: float = .5, epsilon: float = 1e-6,
    unrated_player_rate: float = 1500., conversion_std: float = 173.7178
) -> "list[Glicko2Rate]":

    """Rates players by calculating their new glicko2 after a set of
    interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5). Interactions
    in different formats are converted automatically if possible.

    See also: :meth:`poprank.functional.glicko.glicko`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
            get a rating from. Every interaction should be between exactly 2
            players.
        ratings (list[Glicko2Rate]): the initial ratings of the players.
        RD_unrated (float, optional): The rating deviation of unrated players.
            Defaults to 350.0.
        tau (float, optional): Constant constraining the volatility over time.
            Defaults to 0.5.
        epsilon (float, optional): treshold of tolerance for the iterative
            algorithm. Defaults to 1e-6.
        new_player_rate (float, optional): rating for new players.
            Defaults to 1500.0.
        conversion_var (float): conversion factor between normal and glicko2
            scale. Defaults to 173.7178.

    Returns:
        list[Glicko2Rate]: the updated ratings of all players
    """

    def f(x: float, volatility: float, delta: float,
          std: float, v: float, tau: float) -> float:
        a = (1/2) * (exp(x)*(delta**2 - std**2 - v - exp(x)) /
                     (std**2 + v + exp(x))**2)
        b = (x - log(volatility**2))/tau**2
        return a - b

    # Convert the ratings into Glicko-2 scale
    new_ratings: "list[Glicko2Rate]" = [deepcopy(r) for r in ratings]
    new_ratings = [r._replace(mu=(r.mu - unrated_player_rate)/conversion_std,
                              std=r.std/conversion_std) for r in new_ratings]

    # Initialize values
    variance: "list[float]" = [0. for p in players]
    estimated_improvement: "list[float]" = [0. for p in players]
    sum_match: "list[float]" = [0. for p in players]

    for interaction in interactions:
        id_player: int = players.index(interaction.players[0])
        id_opponent: int = players.index(interaction.players[1])

        # Convert outcomes into (1, 0), (0, 1), (.5, .5) format
        if interaction.outcomes[0] > interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (1, 0)
        elif interaction.outcomes[0] < interaction.outcomes[1]:
            match_outcome: "tuple[float]" = (0, 1)
        else:
            match_outcome: "tuple[float]" = (.5, .5)

        reduce_impact_player: float = \
            Glicko2Rate.reduce_impact(new_ratings[id_opponent].std)
        reduce_impact_opponent: float = \
            Glicko2Rate.reduce_impact(new_ratings[id_player].std)

        expected_outcome_player: float = \
            new_ratings[id_player].expected_outcome(
                new_ratings[id_opponent])
        expected_outcome_opponent: float = \
            new_ratings[id_opponent].expected_outcome(
                new_ratings[id_player])

        variance[id_player] += (reduce_impact_player**2) * \
            expected_outcome_player * (1 - expected_outcome_player)
        variance[id_opponent] += (reduce_impact_opponent**2) * \
            expected_outcome_opponent * (1 - expected_outcome_opponent)

        sum_match[id_player] += reduce_impact_player * \
            (match_outcome[0] - expected_outcome_player)
        sum_match[id_opponent] += reduce_impact_opponent * \
            (match_outcome[1] - expected_outcome_opponent)

    # Set variance to -1 for players that did not have any match
    variance = [1/x if x != 0 else -1 for x in variance]
    estimated_improvement = [a*b for a, b in zip(variance, sum_match)]

    for i, rating in enumerate(new_ratings):

        # Set initial values for iterative algorithm
        alpha: float = log(rating.volatility ** 2)

        if estimated_improvement[i]**2 > rating_deviation_unrated**2 + \
           variance[i]:
            b: float = log(estimated_improvement[i]**2 -
                           rating_deviation_unrated**2 -
                           variance[i])
        else:
            k: int = 1
            while f(alpha - k * sqrt(volatility_constraint ** 2),
                    rating.volatility,
                    estimated_improvement[i], rating.std, variance[i],
                    volatility_constraint) < 0:
                k += 1
            b: float = alpha - k * sqrt(volatility_constraint**2)

        fa: float = f(alpha, rating.volatility, estimated_improvement[i],
                      rating.std, variance[i], volatility_constraint)
        fb: float = f(b, rating.volatility, estimated_improvement[i],
                      rating.std, variance[i], volatility_constraint)

        # Iterate

        while abs(b - alpha) > epsilon:
            c: float = alpha + (alpha - b) * fa / (fb - fa)
            fc: float = f(c, rating.volatility, estimated_improvement[i],
                          rating.std, variance[i], volatility_constraint)

            if fc * fb < 0:
                alpha, fa = b, fb
            else:
                fa /= 2

            b, fb = c, fc

        new_volatility: float = exp(alpha/2)
        estimated_std: float = sqrt(rating.std ** 2 + new_volatility**2)

        # If the player did not play during this rating period
        if variance[i] == -1:
            new_std: float = estimated_std
            new_mu = rating.mu
            new_volatility = rating.volatility
        else:
            new_std: float = 1 / sqrt(1/estimated_std**2 + 1/variance[i])
            new_mu: float = rating.mu + new_std**2 * sum_match[i]

        new_ratings[i].volatility = new_volatility
        new_ratings[i] = new_ratings[i]._replace(mu=new_mu * conversion_std +
                                                 unrated_player_rate,
                                                 std=new_std * conversion_std)

    return new_ratings
