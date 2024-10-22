from math import sqrt, log, exp
from typing import List, Union
from math import e, pi

from popcore import Interaction

from poprank import Rate
from poprank.utils import to_pairwise
from .elo import EloRate
from ..math import sigmoid


class GlickoRate(EloRate):
    """Glicko rating

    :param float mu: Player's initial rating. Defaults to 1500.
    :param float std: Player's default standard deviation. Defaults to 350
    """

    def __init__(
        self, mu: float = 1500, std: float = 350,
        base: float = 10.0, spread: float = 400.0,
        time_since_last_competition: int = 0
    ):
        EloRate.__init__(self, mu, std, base, spread)
        self.time_since_last_competition = time_since_last_competition

    def reduce_impact(self, RD_i: float) -> float:
        """Originally g(RDi), reduces the impact of a game based on the
        opponent's rating_deviation


        :param float RD_i: Rating deviation of the opponent
        :param float q: Q constant. Typically ln(10)/400 in glicko1
            but equal to 1 for glicko2
        :return: g(RDi)
        :rtype: float
        """

        # TODO: Check numerical stability
        return 1 / sqrt(1 + (3 * (self.q**2) * (RD_i**2)) / (pi**2))

    def predict(self, opponent_glicko: "GlickoRate") -> float:
        """Calculate the expected outcome of a match in the glicko1 system

        :param GlickoRate opponent_glicko: Opponent's rating
        :return: The expected score.
        :rtype: float
        """
        if not isinstance(opponent_glicko, GlickoRate):
            raise TypeError("opponent_glicko should be of type Glicko1Rate")

        # g_RD_i on the Glicko paper
        scale = self.reduce_impact(opponent_glicko.std)

        return sigmoid(
            scale * (opponent_glicko.mu - self.mu) / self.spread, self.base)

    def __repr__(self) -> str:
        return (
            f"GlickRate("
            f"mu={self.mu}, std={self.std},"
            f"base={self.base}, spread={self.spread},"
            f"ltc={self.time_since_last_competition})"
        )


class Glicko2Rate(GlickoRate):
    """Glicko2 rating.

    :param float mu: Player's initial rating. Defaults to 0.
    :param float std: Player's default standard deviation. Defaults to 1
    :param float base: The base of the exponent in the elo formula.
    Defaults to 10.0.
    :param float spread:The divisor of the exponent in the elo formula.
    Defaults to 400.0.
    """

    def __init__(
        self, mu: float = 0, std: float = 1.0, base: float = 10.0,
        spread: float = 400.0, volatility: float = 0.06,
        time_since_last_competition: int = 0
    ):
        GlickoRate.__init__(
            self, mu, std, base, spread, time_since_last_competition)
        self.volatility = volatility


def _compute_skill_improvement(
    match_outcome: float, player_rating: GlickoRate,
    opponent_rating: GlickoRate
):
    """
    Calculate the rating adjustment for both players based on expected
    outcome, actual outcome and rating deviation
    Factor reducing the impact of games based on opponent rating
    deviation (Higher RD means lower impact)
    """
    reduce_impact = player_rating.reduce_impact(opponent_rating.std)
    expected_outcome = player_rating.predict(opponent_rating)
    skill_improvement = reduce_impact * (
        match_outcome - expected_outcome)
    variance = reduce_impact**2 * expected_outcome * (
        1.0 - expected_outcome)

    return skill_improvement, variance


def _improvements_from_interactions(
    players: List[str], ratings: List[GlickoRate],
    interactions: List[Interaction]
):
    skill_improvements: "List[float]" = [0. for p in players]
    skill_variance: "List[float]" = [0. for p in players]

    for interaction in to_pairwise(interactions):
        player: int = players.index(interaction.players[0])
        opponent: int = players.index(interaction.players[1])

        match_outcome = interaction.outcomes
        skill_improvement, variance = _compute_skill_improvement(
            match_outcome[0], ratings[player], ratings[opponent]
        )

        skill_improvements[player] += skill_improvement
        skill_variance[player] += variance

        skill_improvement, variance = _compute_skill_improvement(
            match_outcome[1], ratings[opponent], ratings[player]
        )

        skill_improvements[opponent] += skill_improvement
        skill_variance[opponent] += variance

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
    players: "List[str]", interactions: "List[Interaction]",
    ratings: "List[GlickoRate]", uncertainty_increase: float = 34.6,
    rating_deviation_unrated: float = 350.0, base: float = 10.0,
    spread: float = 400.0
) -> "List[GlickoRate]":
    """Rates players by calculating their new glicko after a set of
    interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5). Interactions
    in different formats are converted automatically if possible.

    :param List[str] players: A list containing all unique player identifiers
    :param List[Interaction] interactions: A list containing the
        interactions to get a rating from. Every interaction should be between
        exactly 2 players.
    :param List[Glicko1Rate] ratings: The initial ratings of the players.
    :param float uncertainty_increase: Constant governing the
        increase in uncerntainty between rating periods. Defaults to 34.6.
    :param float rating_deviation_unrated: The rating deviation of
        unrated players. Defaults to 350.0.
    :param float base: Value in the logarithm for the constant q.
        Defaults to 10.0.
    :param float spread: Denominator in the constant q.
        Defaults to 400.0.

    :return: The updated ratings of all players
    :rtype: List[Glicko1Rate]

    Example
    -------

    .. code-block:: python

        # Example from Glickman's paper
        # http://www.glicko.net/glicko/glicko.pdf

        from poprank.functional.glicko import glicko
        from poprank import GlickoRate
        from popcore import Interaction

        players = ["a", "b", "c", "d"]
        interactions = [
            Interaction(["a", "b"], [1, 0]),
            Interaction(["a", "c"], [0, 1]),
            Interaction(["a", "d"], [0, 1]),
            Interaction(["b", "c"], [0, 1]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [.5, .5])
        ]
        ratings = [
            GlickoRate(1500, 200), GlickoRate(1400, 30),
            GlickoRate(1550, 100), GlickoRate(1700, 300)
        ]

        g_results = glicko(players, interactions, ratings)
        g_results = [
            GlickoRate(round(x.mu, 3), round(x.std, 3))
            for x in g_results
        ]

        # g_results is
        # [GlickoRate(1464.106, 151.399), GlickoRate(1396.046, 29.800),
        # GlickoRate(1588.344, 92.598), GlickoRate(1742.969, 194.514)]

    .. seealso::
        :meth:`poprank.functional.glicko2`

        :class:`poprank.rates.GlickoRate`
    """

    next_rating: "List[GlickoRate]" = []

    def convert_to_glicko_rate(elo: Union[float, Rate, EloRate]):
        if not isinstance(elo, GlickoRate):
            if isinstance(elo, float):
                return GlickoRate(mu=elo, base=base, spread=spread)
            if isinstance(elo, Rate):
                return Glicko2Rate(mu=elo.mu, base=base, spread=spread)
            else:
                raise TypeError(elo)
        return elo

    ratings = list(map(convert_to_glicko_rate, ratings))

    # Update rating deviations
    for rating in ratings:
        default_std = min(
            sqrt(
                rating.std ** 2 +
                rating.time_since_last_competition * uncertainty_increase**2
            ),
            rating_deviation_unrated
        )
        next_rating.append(
            GlickoRate(
                mu=rating.mu, std=default_std
            )
        )

    q: float = log(base) / spread

    skill_improvements, skill_variances = _improvements_from_interactions(
        players, next_rating, interactions
    )

    for idx, rating in enumerate(players):
        # Only update if the player had interactions
        if skill_variances[idx] is not None:
            # NOTE: product of two Gaussians?
            new_variance = 1.0 / next_rating[idx].std ** 2
            new_variance += 1.0 / skill_variances[idx]
            new_rating = next_rating[idx].mu
            new_rating += q / new_variance * skill_improvements[idx]
            next_rating[idx].mu = new_rating
            next_rating[idx].std = sqrt(1.0 / new_variance)

    return next_rating


def glicko2(
    players: "List[str]", interactions: "List[Interaction]",
    ratings: "List[Glicko2Rate]", rating_deviation_unrated: float = 350.0,
    volatility_constraint: float = 0.5, epsilon: float = 1e-6,
    unrated_player_rate: float = 1500.0, conversion_std: float = 173.7178,
    base: float = 10.0, spread: float = 400.0
) -> "List[Glicko2Rate]":

    """Rates players by calculating their new glicko2 after a set of
    interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5). Interactions
    in different formats are converted automatically if possible.

    :param List[str] players: A list containing all unique player identifiers
    :param List[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly 2
        players.
    :param List[Glicko2Rate] ratings: The initial ratings of the players.
    :param float RD_unrated: The rating deviation of unrated players.
        Defaults to 350.0.
    :param float tau: Constant constraining the volatility over time.
        Defaults to 0.5.
    :param float epsilon: Treshold of tolerance for the iterative
        algorithm. Defaults to 1e-6.
    :param float new_player_rate: Rating for new players.
        Defaults to 1500.0.
    :param float conversion_var: Conversion factor between normal and glicko2
        scale. Defaults to 173.7178.

    :return: The updated ratings of all players
    :rtype: list[Glicko2Rate]

    Example
    -------

    .. code-block:: python

        # Example from Glickman's paper
        # http://www.glicko.net/glicko/glicko2.pdf

        from poprank.functional.glicko import glicko2
        from poprank import Glicko2Rate
        from popcore import Interaction

        players = ["a", "b", "c", "d"]
        interactions = [
            Interaction(["a", "b"], [1, 0]),
            Interaction(["a", "c"], [0, 1]),
            Interaction(["a", "d"], [0, 1]),
            Interaction(["b", "c"], [0, 1]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [.5, .5])
        ]
        ratings = [
            Glicko2Rate(1500, 200), Glicko2Rate(1400, 30),
            Glicko2Rate(1550, 100), Glicko2Rate(1700, 300)]
        tau = 0.5

        g_results = glicko2(players, interactions, ratings, tau)

        # g_results (rounded to 3 digits) is equal to
        # [Glicko2Rate(1464.051, 151.517), Glicko2Rate(1395.575, 31.522),
        # Glicko2Rate(1588.701, 93.027), Glicko2Rate(1742.991, 194.563)]

        self.assertListEqual(g_results, expected_results)

    .. seealso::
        :meth:`poprank.functional.glicko`

        :class:`poprank.rates.GlickoRate`
    """

    def estimate_volatility(
        rating: Glicko2Rate, improvement: float, variance: float
    ):
        # Step 5: determine new volatility value using Illinois algorithm
        def f(x: float, volatility: float, delta: float,
              std: float, v: float, tau: float) -> float:
            a = exp(x)*(delta**2 - std**2 - v - exp(x))
            a /= (std**2 + v + exp(x))**2
            a *= 0.5
            b = (x - log(volatility**2)) / tau**2
            return a - b

        alpha: float = log(rating.volatility ** 2)
        if improvement ** 2 > rating_deviation_unrated ** 2 + variance:
            b: float = log(
                improvement ** 2 - rating_deviation_unrated ** 2 - variance)
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

    def convert_to_glicko_rate(elo: Union[float, Rate, EloRate]):
        if not isinstance(elo, Glicko2Rate):
            if isinstance(elo, float):
                return Glicko2Rate(mu=elo, base=base, spread=spread)
            if isinstance(elo, Rate):
                return Glicko2Rate(mu=elo.mu, base=base, spread=spread)
            else:
                raise TypeError(elo)
        return elo

    ratings = list(map(convert_to_glicko_rate, ratings))

    base = ratings[0].base
    spread = ratings[0].spread

    assert all([(r.base, r.spread) == (base, spread) for r in ratings])

    # Convert the ratings into Glicko-2 scale
    next_rating = [
        Glicko2Rate(
            mu=(r.mu - unrated_player_rate) / conversion_std,
            std=r.std / conversion_std,
            base=e,
            spread=1.0)
        for r in ratings
    ]

    skill_improvements, skill_variances = _improvements_from_interactions(
        players, next_rating, interactions
    )

    for idx, rating in enumerate(next_rating):
        if skill_variances[idx] is None:
            # if player did not played on the interactions
            new_std = sqrt(rating.std ** 2 + rating.volatility**2)
            new_mu = rating.mu
            new_volatility = rating.volatility
        else:
            new_volatility = estimate_volatility(
                next_rating[idx],
                improvement=skill_improvements[idx] * skill_variances[idx],
                variance=skill_variances[idx]
            )
            estimated_std: float = sqrt(rating.std ** 2 + new_volatility**2)
            new_variance = 1.0 / estimated_std**2 + 1.0 / skill_variances[idx]
            new_std = 1.0 / sqrt(new_variance)
            new_mu = rating.mu + new_std**2 * skill_improvements[idx]

        next_rating[idx].volatility = new_volatility
        next_rating[idx].mu = new_mu * conversion_std + unrated_player_rate
        next_rating[idx].std = new_std * conversion_std
        next_rating[idx].base = base
        next_rating[idx].spread = spread

    return next_rating
