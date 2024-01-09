from math import log

from ..elo import EloRate
from .data import BayesEloStats, PairwiseStatistics


class BayesEloRating:
    """Rates players by calculating their new elo using a bayeselo approach
    Given a set of interactions and initial elo ratings, uses a
    Minorization-Maximization algorithm to estimate maximum-likelihood
    ratings.
    Made to imitate https://www.remi-coulom.fr/Bayesian-Elo/

    Args:
        pairwise_stats (PopulationPairwisestatistics): The summary
            of all interactions between players
        elos (list[EloRate]): The ititial ratings of the players
        elo_advantage (float, optional): The home-field-advantage
            expressed as rating points. Defaults to 32.8.
        elo_draw (float, optional): The probability of drawing.
            Defaults to 97.3.
        base (float, optional): The base of the exponent in the elo
            formula. Defaults to 10.0
        spread (float, optional): The divisor of the exponent in the elo
            formula. Defaults to 400.0.
        home_field_bias (float, optional): _description_. Defaults to 0.0.
        draw_bias (float, optional): _description_. Defaults to 0.0.

    Methods:
        update_ratings(self) -> None: Performs one iteration of the
            Minorization-Maximization algorithm
        update_home_field_bias(self) -> float: Use interaction statistics
            to update the home_field_bias automatically
        update_draw_bias(self) -> float: Use interaction statistics to
            update the draw_bias automatically
        compute_difference(self, ratings: "list[float]",
            next_ratings: "list[float]") -> float: Compute the impact of
                the current interation on ratings
        minorize_maximize(self, learn_home_field_bias: bool,
            home_field_bias: float, learn_draw_bias: bool,
            draw_bias: float, iterations: int, tolerance: float
            ) -> None: Perform the MM algorithm for generalized
                Bradley-Terry models.
    """

    def __init__(
        self, pairwise_stats: BayesEloStats,
        elos: "list[EloRate]", elo_advantage: float = 32.8,
        elo_draw: float = 97.3, base=10., spread=400.,
        home_field_bias=0.0, draw_bias=0.0
    ):

        # Condensed results
        self.pairwise_stats: BayesEloStats = pairwise_stats
        self.elos = elos  # Players elos
        self.elo_advantage = elo_advantage  # advantage of playing white
        self.elo_draw = elo_draw  # likelihood of drawing
        self.ratings = [0. for x in range(pairwise_stats.num_players)]
        self.next_ratings = [0. for x in range(pairwise_stats.num_players)]
        self.base = base
        self.spread = spread
        self.home_field_bias: float = home_field_bias
        self.draw_bias: float = draw_bias

    def update_ratings(self) -> None:
        """Performs one iteration of the Minorization-Maximization algorithm"""
        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            A: float = 0.0
            B: float = 0.0

            for opponent in range(
              self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result: PairwiseStatistics = \
                    self.pairwise_stats.statistics[player][opponent]

                if result.opponent_idx > player:
                    opponent_rating = self.next_ratings[result.opponent_idx]
                else:
                    opponent_rating = self.ratings[result.opponent_idx]

                A += result.w_ij + result.d_ij + result.l_ji + result.d_ji

                B += ((result.d_ij + result.w_ij) * self.home_field_bias /
                      (self.home_field_bias * self.ratings[player] +
                      self.draw_bias * opponent_rating) +
                      (result.d_ij + result.l_ij) * self.draw_bias *
                      self.home_field_bias /
                      (self.draw_bias * self.home_field_bias *
                       self.ratings[player] +
                      opponent_rating) +
                      (result.d_ji + result.w_ji) * self.draw_bias /
                      (self.home_field_bias * opponent_rating +
                      self.draw_bias * self.ratings[player]) +
                      (result.d_ji + result.l_ji) /
                      (self.draw_bias * self.home_field_bias *
                       opponent_rating +
                      self.ratings[player]))

            self.next_ratings[player] = A / B

        self.ratings, self.next_ratings = self.next_ratings, self.ratings

    def update_home_field_bias(self) -> float:
        """Use interaction statistics to update the home_field_bias
        automatically"""
        numerator: float = 0.
        denominator: float = 0.

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            for opponent in range(
              self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result = self.pairwise_stats.statistics[player][opponent]
                opponent_rating = self.ratings[result.opponent_idx]

                numerator += result.w_ij + result.d_ij
                denominator += ((result.d_ij + result.w_ij) *
                                self.ratings[player] /
                                (self.home_field_bias * self.ratings[player] +
                                self.draw_bias * opponent_rating) +
                                (result.d_ij + result.l_ij) * self.draw_bias *
                                self.ratings[player] /
                                (self.draw_bias * self.home_field_bias *
                                self.ratings[player] + opponent_rating))

        return numerator / denominator

    def update_draw_bias(self) -> float:
        """Use interaction statistics to update the draw_bias automatically"""
        numerator: float = 0.
        denominator: float = 0.

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            for opponent in range(
              self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result = self.pairwise_stats.statistics[player][opponent]
                opponent_rating = self.ratings[result.opponent_idx]

                numerator += result.d_ij
                denominator += ((result.d_ij + result.w_ij) * opponent_rating /
                                (self.home_field_bias * self.ratings[player] +
                                self.draw_bias * opponent_rating) +
                                (result.d_ij + result.l_ij) *
                                self.home_field_bias *
                                self.ratings[player] /
                                (self.draw_bias * self.home_field_bias *
                                self.ratings[player] + opponent_rating))

        c: float = numerator / denominator
        return c + (c * c + 1)**0.5

    def compute_difference(self, ratings: "list[float]",
                           next_ratings: "list[float]") -> float:
        """Compute the impact of the current interation on ratings"""
        return max([abs(a-b)/(a+b) for a, b in zip(ratings, next_ratings)])

    def minorize_maximize(
        self,
        learn_home_field_bias: bool = False,
        home_field_bias: float = 1.,
        learn_draw_bias: bool = False,
        draw_bias: float = 1.,
        iterations: int = 10000,
        tolerance: float = 1e-5,
    ) -> None:
        """Perform the MM algorithm for generalized Bradley-Terry models.

        The Minorization-Maximization algorithm is performed for the number of
        specified iterations or until the changes are below the tolerance
        value, whichever comes first.
        Args:
            use_home_field_bias (bool, optional): _description_. Defaults to
                False.
            home_field_bias (float, optional): _description_. Defaults to 1.0.
            learn_draw_bias (bool, optional): _description_. Defaults to False.
            draw_bias (float, optional): _description_. Defaults to 1.0.
            iterations (int, optional): _description_. Defaults to 10000.
            tolerance (float, optional): _description_. Defaults to 1e-5.
        """

        # Set initial values
        self.home_field_bias = home_field_bias
        self.draw_bias = draw_bias
        self.ratings = [1. for p in range(self.pairwise_stats.num_players)]

        # Main MM loop
        for player in range(iterations):
            self.update_ratings()
            diff = self.compute_difference(self.ratings, self.next_ratings)

            if learn_home_field_bias:
                new_home_field_bias = self.update_home_field_bias()
                home_field_bias_diff = \
                    abs(self.home_field_bias - new_home_field_bias)
                if home_field_bias_diff > diff:
                    diff = home_field_bias_diff
                self.home_field_bias = new_home_field_bias

            if learn_draw_bias:
                new_draw_bias = self.update_draw_bias()
                draw_bias_diff = abs(self.draw_bias - new_draw_bias)
                if draw_bias_diff > diff:
                    diff = draw_bias_diff
                self.draw_bias = new_draw_bias

            if diff < tolerance:
                break

        # Convert back to Elos
        total: float = \
            sum([log(self.ratings[player], self.base) * self.spread
                 for player in range(self.pairwise_stats.num_players)])

        offset: float = -total / self.pairwise_stats.num_players

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            self.elos[player].mu = log(
                self.ratings[player], self.base) * self.spread + offset

        if learn_home_field_bias:
            self.elo_advantage = \
                log(self.home_field_bias, self.base) * self.spread
        if learn_draw_bias:
            self.elo_draw = log(self.draw_bias, self.base) * self.spread

    def rescale_elos(self) -> None:
        """Rescales the elos by a common factor"""
        # EloScale
        for i, e in enumerate(self.elos):
            x: float = e.base**(-self.elo_draw/e.spread)
            elo_scale: float = x * 4.0 / ((1 + x) ** 2)
            tmp_base: float = self.elos[i].base
            tmp_spread: float = self.elos[i].spread
            self.elos[i]: EloRate = EloRate(
                self.elos[i].mu * elo_scale,
                self.elos[i].std
            )
            self.elos[i].base = tmp_base
            self.elos[i].spread = tmp_spread
