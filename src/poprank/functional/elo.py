from math import log
from dataclasses import dataclass
from popcore import Interaction
from poprank import Rate, EloRate
from poprank.functional.wdl import windrawlose


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float, wdl: bool = False
) -> "list[EloRate]":
    """Rates players by calculating their new elo after a set of interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5).

    See also: :meth:`poprank.functional.bayeselo`

    Args:
        players (list[str]): a list containing all unique player identifiers
        interactions (list[Interaction]): a list containing the interactions to
            get a rating from. Every interaction should be between exactly 2
            players and result in a win (1, 0), a loss (0, 1)
            or a draw (0.5, 0.5)
        elos (list[EloRate]): the initial ratings of the players
        k_factor (float): maximum possible adjustment per game. Larger means
            player rankings change faster
        wdl (bool, optional): Turn the interactions into the (1, 0), (.5, .5),
            (0, 1) format automatically. Defaults to False.
    Raises:
        ValueError: if the numbers of players and ratings don't match,
            if an interaction has the wrong number of players,
            if an interaction has the wrong number of outcomes,
            if a player that does not appear in `players`is in an
            interaction
        TypeError: Using Rate instead of EloRate

    Returns:
        list[EloRate]: the updated ratings of all players
    """

    # Checks
    if len(players) != len(elos):
        raise ValueError(f"Players and elos length mismatch\
: {len(players)} != {len(elos)}")

    for elo in elos:
        if not isinstance(elo, EloRate):
            raise TypeError("elos must be of type list[EloRate]")

    for interaction in interactions:
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Elo only accepts interactions involving \
both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player \
list")

        if not wdl and (interaction.outcomes[0] not in (0, .5, 1) or
                        interaction.outcomes[1] not in (0, .5, 1) or
                        sum(interaction.outcomes) != 1):
            raise Warning("Elo takes outcomes in the (1, 0), (0, 1), (.5, .5) \
format, other values may have unspecified behavior \
(set wdl=True to automatically turn interactions \
into the windrawlose format)")

    # Calculate the expected score vs true score of all players in the given
    # set of interactions and adjust elo afterwards accordingly.
    expected_scores: "list[float]" = [.0 for player in players]
    true_scores: "list[float]" = [.0 for player in players]

    for interaction in interactions:
        id_p1 = players.index(interaction.players[0])
        id_p2 = players.index(interaction.players[1])

        expected_scores[id_p1] += elos[id_p1].expected_outcome(elos[id_p2])
        expected_scores[id_p2] += elos[id_p2].expected_outcome(elos[id_p1])

        true_scores[players.index(interaction.players[0])] += \
            interaction.outcomes[0]
        true_scores[players.index(interaction.players[1])] += \
            interaction.outcomes[1]

    if wdl:
        true_scores = [r.mu for r in
                       windrawlose(players=players,
                                   interactions=interactions,
                                   ratings=[Rate(0, 0) for p in players],
                                   win_value=1,
                                   draw_value=.5,
                                   loss_value=0)]
    # New elo values
    rates: "list[EloRate]" = \
        [EloRate(e.mu + k_factor*(true_scores[i] - expected_scores[i]), 0)
         for i, e in enumerate(elos)]

    return rates


@dataclass
class PairwiseStatistics:  # cr
    """A condensed summary of all the interactions between two players.

    Args:
        player_idx (int, optional): Id of the player. Defaults to -1.
        opponent_idx (int, optional): Id of the opponent. Defaults to -1.
        total_games (int, optional): Total number of games played.
            Defaults to 0.
        w_ij (float, optional):  # wins of player i against opponent j.
            Defaults to 0.
        d_ij (float, optional):  # draws of player i against opponent j.
            Defaults to 0.
        l_ij (float, optional):  # losses of player i against opponent j.
            Defaults to 0.
        w_ji (float, optional):  # wins of opponent j against player i.
            Defaults to 0.
        d_ji (float, optional):  # draws of opponent j against player i.
            Defaults to 0.
        l_ji (float, optional):  # losses of opponent j against player i.
            Defaults to 0.
    """

    player_idx: int = -1  # id of the player
    opponent_idx: int = -1  # id of the opponent
    total_games: int = 0  # Total number of games played
    w_ij: float = 0  # win player i against player j
    d_ij: float = 0  # draw player i against player j
    l_ij: float = 0  # loss player i against player j
    w_ji: float = 0  # win player j against player i
    d_ji: float = 0  # draw player j against player i
    l_ji: float = 0  # loss player j against player i


def count_total_opponent_games(
        player_idx: int,
        statistics: "list[list[PairwiseStatistics]]") -> int:
    """Return the sum of all games played by opponents of the player

    Args:
        player_idx (int): Id of the player
        num_opponents_per_player (list[int]): Number of opponents per player
        statistics (list[list[PairwiseStatistics]]): The array of pairwise
            statistics
    """
    return sum([opponent.total_games for opponent in statistics[player_idx]])


def find_opponent(player_idx: int,
                  opponent_idx: int,
                  num_opponents_per_player: "list[int]",
                  statistics: "list[list[PairwiseStatistics]]"
                  ) -> PairwiseStatistics:
    """Return the pairwise interaction statistics between the player and the
    opponent

    Args:
        player_idx (int): Id of the player
        opponent_idx (int): Id of the opponent
        num_opponents_per_player (list[int]): Number of opponents per player
        statistics (list[list[PairwiseStatistics]]): The array of pairwise
            statistics

    Raises:
        RuntimeError: If the opponent could not be foud
    """
    for x in range(num_opponents_per_player[player_idx]):
        if statistics[player_idx][x].opponent_idx == opponent_idx:
            return statistics[player_idx][x]
    raise RuntimeError(f"Cound not find opponent {opponent_idx} \
                       for player {player_idx}")


@dataclass
class PopulationPairwiseStatistics:  # crs
    """The pairwise statistics of an entire population

    Args:
        num_players(int): Number of players in the population
        num_opponents_per_players (list[int]): Number of opponents for
            each player
        statistics (list[list[PairwiseStatistics]]): Results for each
            pair of players

    Static Methods:
        from_interactions(
            players: 'list[str]',
            interactions: 'list[Interaction]',
            add_draw_prior: bool = True,
            draw_prior: float = 2.0
            ) -> 'PopulationPairwiseStatistics': Turn a list of interactions
                into pairwise statistics

    Instance Methods:
         add_opponent(
            player: str,
            opponent: str,
            ppcr_ids: "list[int]",
            indx: "dict[str, int]"
        ) -> None: Add an opponent to the player

        def add_prior(draw_prior: float = 2.0) -> None:
            Add prior draws to pairwise statistics
    """
    num_players: int  # Number of players in the pop
    num_opponents_per_player: "list[int]"  # nbr of opponents for each player
    statistics: "list[list[PairwiseStatistics]]"  # Results for each match

    def add_opponent(
        self,
        player: str,
        opponent: str,
        ppcr_ids: "list[int]",
        indx: "dict[str, int]"
    ) -> None:
        """Add an opponent to the player"""
        ppcr_ids[indx[player]].append(opponent)
        self.statistics[indx[player]].append(PairwiseStatistics(
            player_idx=indx[player],
            opponent_idx=indx[opponent]
        ))
        self.num_opponents_per_player[indx[player]] += 1

    def add_prior(self, draw_prior: float = 2.0) -> None:
        """Add prior draws to pairwise statistics"""
        for player, _ in enumerate(self.statistics):
            prior: float = draw_prior * 0.25 / count_total_opponent_games(
                player, self.statistics)

            for opponent in range(self.num_opponents_per_player[player]):
                cr_player = self.statistics[player][opponent]
                cr_opponent = find_opponent(
                    cr_player.opponent_idx, player,
                    self.num_opponents_per_player, self.statistics)
                this_prior: float = prior * cr_player.total_games
                cr_player.d_ij += this_prior
                cr_player.d_ji += this_prior
                cr_opponent.d_ij += this_prior
                cr_opponent.d_ji += this_prior

    @staticmethod
    def from_interactions(
        players: 'list[str]',
        interactions: 'list[Interaction]',
        add_draw_prior: bool = True,
        draw_prior: float = 2.0
    ) -> 'PopulationPairwiseStatistics':
        """Turn a list of interactions into pairwise statistics

        Args:
            players (list[str]): The list of players
            interactions (list[Interaction]): The list of interactions to
                turn into pairwise statistics.
            add_draw_prior (bool, optional): If true, draws will be added to
                pairwise statistics to avoid division by zero errors.
                Defaults to True.
            draw_prior (float, optional): Value of the draws to add.
                Defaults to 2.0.
        """

        # have fun figuring out this indexing mess :)
        num_opponents_per_player: "list[int]" = [0 for p in players]
        statistics: "list[list[PairwiseStatistics]]" = [[] for p in players]
        ppcr_ids: "list[list[str]]" = [[] for p in players]
        indx: "dict[str, int]" = {p: i for i, p in enumerate(players)}

        pps: PopulationPairwiseStatistics = PopulationPairwiseStatistics(
            num_players=len(players),
            num_opponents_per_player=num_opponents_per_player,
            statistics=statistics
        )

        for i in interactions:

            # If the players have never played together before
            if i.players[1] not in ppcr_ids[indx[i.players[0]]]:
                # Add player 1 to the list of opponents of player 0
                pps.add_opponent(i.players[0], i.players[1], ppcr_ids, indx)

                # Add player 0 to the list of opponents of player 1
                pps.add_opponent(i.players[1], i.players[0], ppcr_ids, indx)

            p1_relative_id = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            p0_relative_id = ppcr_ids[indx[i.players[1]]].index(i.players[0])

            if i.outcomes[0] > i.outcomes[1]:  # White wins
                pps.statistics[indx[i.players[0]]][p1_relative_id].w_ij += 1
                pps.statistics[indx[i.players[1]]][p0_relative_id].w_ji += 1

            elif i.outcomes[0] < i.outcomes[1]:  # Black wins
                pps.statistics[indx[i.players[0]]][p1_relative_id].l_ij += 1
                pps.statistics[indx[i.players[1]]][p0_relative_id].l_ji += 1

            else:  # Draw
                pps.statistics[indx[i.players[0]]][p1_relative_id].d_ij += 1
                pps.statistics[indx[i.players[1]]][p0_relative_id].d_ji += 1

            # Update total games
            pps.statistics[indx[i.players[0]]][p1_relative_id].total_games += 1
            pps.statistics[indx[i.players[1]]][p0_relative_id].total_games += 1

        if add_draw_prior:
            pps.add_prior(draw_prior)

        return pps


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
        self, pairwise_stats: PopulationPairwiseStatistics,
        elos: "list[EloRate]", elo_advantage: float = 32.8,
        elo_draw: float = 97.3, base=10., spread=400.,
        home_field_bias=0.0, draw_bias=0.0
    ):

        # Condensed results
        self.pairwise_stats: PopulationPairwiseStatistics = pairwise_stats
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
            self.elos[player] = self.elos[player]._replace(
                mu=log(self.ratings[player], self.base) * self.spread + offset)

        if learn_home_field_bias:
            self.elo_advantage = \
                log(self.home_field_bias, self.base) * self.spread
        if learn_draw_bias:
            self.elo_draw = log(self.draw_bias, self.base) * self.spread

    def rescale_elos(self) -> None:
        """Rescales the elos by a common factor"""
        # EloScale # TODO: Figure out what on earth that is
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


def bayeselo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", elo_base: float = 10., elo_spread: float = 400.,
    elo_draw: float = 97.3, elo_advantage: float = 32.8,
    iterations: int = 10000, tolerance: float = 1e-5
) -> "list[EloRate]":
    """Rates players by calculating their new elo using a bayeselo approach

    Given a set of interactions and initial elo ratings, uses a
    Minorization-Maximization algorithm to estimate maximum-likelihood
    ratings.
    The Minorization-Maximization algorithm is performed for the number of
    specified iterations or until the changes are below the tolerance
    value, whichever comes first.
    Made to imitate https://www.remi-coulom.fr/Bayesian-Elo/

    Args:
        players (list[str]): The list of all players
        interactions (list[Interactions]): The list of all interactions
        elos (list[EloRate]): The initial ratings of the players
        elo_base (float, optional): The base of the exponent in the elo
            formula. Defaults to 10.0
        elo_spread (float, optional): The divisor of the exponent in the elo
            formula. Defaults to 400.0.
        elo_draw (float, optional): The probability of drawing.
            Defaults to 97.3.
        elo_advantage (float, optional): The home-field-advantage
            expressed as rating points. Defaults to 32.8.
        iterations (int, optional): The maximum number of iterations the
            Minorization-Maximization algorithm will go through.
            Defaults to 10000.
        tolerance (float, optional): The error threshold below which the
            Minorization-Maximization algorithm stopt. Defaults to 1e-5.
    """

    if len(players) != len(elos):
        raise ValueError(f"Players and elos length mismatch\
: {len(players)} != {len(elos)}")

    for elo in elos:
        if not isinstance(elo, EloRate):
            raise TypeError("elos must be of type list[EloRate]")

    for interaction in interactions:
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Bayeselo only accepts interactions involving \
both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player \
list")

        if interaction.outcomes[0] not in (0, .5, 1) or \
           interaction.outcomes[1] not in (0, .5, 1) or \
           sum(interaction.outcomes) != 1:
            raise Warning("Bayeselo takes outcomes in the (1, 0), (0, 1), \
(.5, .5) format, other values may have unspecified behavior")

    for e in elos:
        if e.base != elo_base or e.spread != elo_spread:
            raise ValueError(f"Elos with different bases and \
spreads are not compatible (expected base {elo_base}, spread {elo_spread} but \
got base {e.base}, spread {e.spread})")

    pairwise_stats: PopulationPairwiseStatistics = \
        PopulationPairwiseStatistics.from_interactions(
            players=players,
            interactions=interactions
        )

    bt: BayesEloRating = BayesEloRating(
        pairwise_stats, elos, elo_draw=elo_draw, elo_advantage=elo_advantage,
        base=elo_base, spread=elo_spread
    )

    bt.minorize_maximize(
        learn_home_field_bias=False,
        home_field_bias=elo_base ** (elo_advantage/elo_spread),
        learn_draw_bias=False,
        draw_bias=elo_base ** (elo_draw/elo_spread),
        iterations=iterations,
        tolerance=tolerance
    )

    bt.rescale_elos()

    return bt.elos
