from math import sqrt, log, exp
from typing import Tuple
from popcore import Interaction
from poprank import GlickoRate, Glicko2Rate


def _compute_skill_improvement(
    match_outcome: float, player_rating: GlickoRate,
    opponent_rating: GlickoRate
):
    # Calculate the rating adjustment for both players based on expected
    # outcome, actual outcome and rating deviation

    # Factor reducing the impact of games based on opponent rating
    # deviation (Higher RD means lower impact)
    reduce_impact = player_rating.reduce_impact(opponent_rating.std)
    expected_outcome = player_rating.expected_outcome(opponent_rating)
    skill_improvement = reduce_impact * (
        match_outcome - expected_outcome)
    variance = reduce_impact**2 * expected_outcome * (
        1.0 - expected_outcome)

    return skill_improvement, variance


def _interaction_to_match_outcome(interaction: Interaction) -> Tuple[float]:
    if interaction.outcomes[0] > interaction.outcomes[1]:
        match_outcome: "tuple[float]" = (1, 0)
    elif interaction.outcomes[0] < interaction.outcomes[1]:
        match_outcome: "tuple[float]" = (0, 1)
    else:
        match_outcome: "tuple[float]" = (.5, .5)

    return match_outcome


def _improvements_from_interactions(
    players: list[str], ratings: list[GlickoRate],
    interactions: list[Interaction]
):
    skill_improvements: "list[float]" = [0. for p in players]
    skill_variance: "list[float]" = [0. for p in players]

    for interaction in interactions:
        id_player: int = players.index(interaction.players[0])
        id_opponent: int = players.index(interaction.players[1])

        match_outcome = _interaction_to_match_outcome(interaction)
        skill_improvement, variance = _compute_skill_improvement(
            match_outcome[0], ratings[id_player], ratings[id_opponent]
        )

        skill_improvements[id_player] += skill_improvement
        skill_variance[id_player] += variance

        skill_improvement, variance = _compute_skill_improvement(
            match_outcome[1], ratings[id_opponent], ratings[id_player]
        )

        skill_improvements[id_opponent] += skill_improvement
        skill_variance[id_opponent] += variance

    # Set d_squared to None if the player did not have any interaction
    for idx, (skill, var) in enumerate(
            zip(skill_improvements, skill_variance)):
        player_rating = ratings[idx]
        if var == 0.0:
            skill_variance[idx] = None
            player_rating.time_since_last_competition += 1
        else:
            skill_variance[idx] = 1 / (player_rating.q ** 2 * var)

    return skill_improvements, skill_variance  # estimated mean and variances


def glicko(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[GlickoRate]", uncertainty_increase: float = 34.6,
    rating_deviation_unrated: float = 350.0, base: float = 10.0,
    spread: float = 400.0
) -> "list[GlickoRate]":
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

    new_ratings: "list[GlickoRate]" = []

    # Update rating deviations
    for rating in ratings:
        default_std = min(
            sqrt(
                rating.std ** 2 +
                rating.time_since_last_competition * uncertainty_increase**2
            ),
            rating_deviation_unrated
        )
        new_ratings.append(
            GlickoRate(
                mu=rating.mu, std=default_std
            )
        )

    q: float = log(base) / spread

    skill_improvements, skill_variances = _improvements_from_interactions(
        players, new_ratings, interactions
    )

    for idx, rating in enumerate(players):
        # Only update if the player had interactions
        if skill_variances[idx] is not None:
            # NOTE: product of two Gaussians?
            new_variance = 1.0 / new_ratings[idx].std ** 2
            new_variance += 1.0 / skill_variances[idx]
            new_rating = new_ratings[idx].mu
            new_rating += q / new_variance * skill_improvements[idx]
            new_ratings[idx].mu = new_rating
            new_ratings[idx].std = sqrt(1.0 / new_variance)

    return new_ratings


def glicko2(
    players: "list[str]", interactions: "list[Interaction]",

    ratings: "list[Glicko2Rate]", rating_deviation_unrated: float = 350.0,
    volatility_constraint: float = 0.5, epsilon: float = 1e-6,
    unrated_player_rate: float = 1500.0, conversion_std: float = 173.7178
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

    def estimate_volatility(
        rating: Glicko2Rate, improvement: float, variance: float
    ):
        # Step 5: determine new volatility value using Illinois algorithm
        def f(x: float, volatility: float, delta: float,
              std: float, v: float, tau: float) -> float:
            a = exp(x)*(delta**2 - std**2 - v - exp(x)) / (std**2 + v + exp(x))**2
            a *= 0.5
            b = (x - log(volatility**2)) / tau**2
            return a - b

        alpha: float = log(rating.volatility ** 2)
        if improvement**2 > rating_deviation_unrated**2 + variance:
            b: float = log(
                improvement**2 - rating_deviation_unrated**2 - variance)
        else:
            k: int = 1
            while f(alpha - k * sqrt(volatility_constraint ** 2),
                    rating.volatility,
                    improvement, rating.std, variance,
                    volatility_constraint) < 0:
                k += 1
            b: float = alpha - k * sqrt(volatility_constraint**2)

        fa: float = f(
            alpha, rating.volatility, improvement,
            rating.std, variance, volatility_constraint
        )
        fb: float = f(
            b, rating.volatility, improvement,
            rating.std, variance, volatility_constraint
        )

        # Iterate

        while abs(b - alpha) > epsilon:
            c: float = alpha + (alpha - b) * fa / (fb - fa)
            fc: float = f(
                c, rating.volatility, improvement,
                rating.std, variance, volatility_constraint
            )

            if fc * fb < 0:
                alpha, fa = b, fb
            else:
                fa /= 2

            b, fb = c, fc

        return exp(0.5 * alpha)

    # Convert the ratings into Glicko-2 scale
    new_ratings = [
        Glicko2Rate(
            mu=(r.mu - unrated_player_rate) / conversion_std,
            std=r.std / conversion_std)
        for r in ratings
    ]

    skill_improvements, skill_variances = _improvements_from_interactions(
        players, new_ratings, interactions
    )

    for idx, rating in enumerate(new_ratings):
        if skill_variances[idx] is None:
            # if player did not played on the interactions
            new_std = sqrt(rating.std ** 2 + rating.volatility**2)
            new_mu = rating.mu
            new_volatility = rating.volatility
        else:
            new_volatility = estimate_volatility(
                new_ratings[idx],
                improvement=skill_improvements[idx] * skill_variances[idx],
                variance=skill_variances[idx]
            )
            estimated_std: float = sqrt(rating.std ** 2 + new_volatility**2)
            new_variance = 1.0 / estimated_std**2 + 1.0 / skill_variances[idx]
            new_std = 1.0 / sqrt(new_variance)
            new_mu = rating.mu + new_std**2 * skill_improvements[idx]

        new_ratings[idx].volatility = new_volatility
        new_ratings[idx].mu = new_mu * conversion_std + unrated_player_rate
        new_ratings[idx].std = new_std * conversion_std

    return new_ratings
