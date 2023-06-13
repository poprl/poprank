from math import log
from popcore import Interaction
from poprank import Rate, EloRate
from poprank.functional.wdl import windrawlose
from dataclasses import dataclass


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
        wdl (bool): Turn the interactions into the (1, 0), (.5, .5), (0, 1)
            format automatically
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
    expected_scores = [.0 for player in players]
    true_scores = [.0 for player in players]

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
    rates = [EloRate(e.mu + k_factor*(true_scores[i] - expected_scores[i]), 0)
             for i, e in enumerate(elos)]

    return rates


@dataclass
class PairwiseStatistics:  # cr
    """A condensed summary of all the interactions between two players

    Args:
        player_idx (int) = -1: Id of the player
        opponent_idx (int) = -1: Id of the opponent
        total_games (int) = 0: True number of games played
        w_ij (float) = 0:  # wins of player i against opponent j
        d_ij (float) = 0:  # draws of player i against opponent j
        l_ij (float) = 0:  # losses of player i against opponent j
        w_ji (float) = 0:  # wins of opponent j against player i
        d_ji (float) = 0:  # draws of opponent j against player i
        l_ji (float) = 0:  # losses of opponent j against player i
    """

    player_idx: int = -1  # id of the player
    opponent_idx: int = -1  # id of the opponent
    total_games: int = 0  # True number of games played
    w_ij: float = 0  # win player i against player j
    d_ij: float = 0  # draw player i against player j
    l_ij: float = 0  # loss player i against player j
    w_ji: float = 0  # win player j against player i
    d_ji: float = 0  # draw player j against player i
    l_ji: float = 0  # loss player j against player i


def count_total_opponent_games(player, num_opponents_per_player,
                               pairwise_stats):
    """Return the sum of all games played by opponents of the player"""
    return sum([opponent.total_games for opponent in pairwise_stats[player]])
    """result = 0
    for i in range(num_opponents_per_player[player]):
        result += pairwise_stats[player][i].total_games
    return result"""


def find_opponent(player, opponent, num_opponents_per_player, statistics):
    for x in range(num_opponents_per_player[player]):
        if statistics[player][x].opponent_idx == opponent:
            return statistics[player][x]
    raise RuntimeError("Cound not find opponent")


@dataclass
class PopulationPairwiseStatistics:  # crs
    num_players: int  # Number of players in the pop
    num_opponents_per_player: "list[int]"  # Number of opponents for each player
    statistics: "list[list[PairwiseStatistics]]"  # Results for each match

    @staticmethod
    def from_interactions(
        players: 'list[str]',
        interactions: 'list[Interaction]',
        add_draw_prior: bool = True,
        draw_prior: float = 2.0
    ) -> 'PopulationPairwiseStatistics':

        num_opponents_per_player = [0 for p in players]
        statistics = [[] for p in players]
        ppcr_ids = [[] for p in players]
        indx = {p: i for i, p in enumerate(players)}

        for i in interactions:

            if i.players[1] not in ppcr_ids[indx[i.players[0]]]:
                # If the players have never played together before

                # Add player 1 to the list of opponents of player 0
                ppcr_ids[indx[i.players[0]]].append(i.players[1])
                statistics[indx[i.players[0]]].append(PairwiseStatistics(
                    opponent_idx=indx[i.players[1]],
                ))
                num_opponents_per_player[indx[i.players[0]]] += 1

                # Add player 0 to the list of opponents of player 1
                ppcr_ids[indx[i.players[1]]].append(i.players[0])
                statistics[indx[i.players[1]]].append(PairwiseStatistics(
                    opponent_idx=indx[i.players[0]],
                ))
                num_opponents_per_player[indx[i.players[1]]] += 1

            if i.outcomes[0] > i.outcomes[1]:  # White wins
                # Update score of player 0
                tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
                statistics[indx[i.players[0]]][tmp].w_ij += 1

                # Update score of player 1
                tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
                statistics[indx[i.players[1]]][tmp].w_ji += 1

            elif i.outcomes[0] < i.outcomes[1]:  # Black wins
                # Update score of player 0
                tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
                statistics[indx[i.players[0]]][tmp].l_ij += 1

                # Update score of player 1
                tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
                statistics[indx[i.players[1]]][tmp].l_ji += 1

            else:  # Draw
                # Update score of player 0
                tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
                statistics[indx[i.players[0]]][tmp].d_ij += 1

                # Update score of player 1
                tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
                statistics[indx[i.players[1]]][tmp].d_ji += 1

            # Update true games of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            statistics[indx[i.players[0]]][tmp].total_games += 1

            # Update true games of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            statistics[indx[i.players[1]]][tmp].total_games += 1

        # addPrior
        if add_draw_prior:
            draw_prior = 2.
            for player, stats in enumerate(statistics):
                prior = draw_prior * 0.25 / count_total_opponent_games(
                    player, num_opponents_per_player, statistics)

                for opponent in range(num_opponents_per_player[player]):
                    crPlayer = statistics[player][opponent]
                    crOpponent = find_opponent(
                        crPlayer.opponent_idx, player,
                        num_opponents_per_player, statistics)
                    thisPrior = prior * crPlayer.total_games
                    crPlayer.d_ij += thisPrior
                    crPlayer.d_ji += thisPrior
                    crOpponent.d_ij += thisPrior
                    crOpponent.d_ji += thisPrior

        return PopulationPairwiseStatistics(
            num_players=len(players),
            num_opponents_per_player=num_opponents_per_player,
            statistics=statistics
        )


class BradleyTerryModel:
    def __init__(
        self, pairwise_stats: PopulationPairwiseStatistics,
        elos: "list[EloRate]", elo_advantage: float = 32.8,
        elo_draw: float = 97.3, base=10, spread=400,
        home_field_bias=0.0, draw_bias=0.0
    ):
        self.pairwise_stats: PopulationPairwiseStatistics = pairwise_stats  # Condensed results
        self.elos = elos  # Players elos
        self.eloAdvantage = elo_advantage  # advantage of playing white
        self.eloDraw = elo_draw  # likelihood of drawing
        self.ratings = [0. for x in range(pairwise_stats.num_players)]
        self.next_ratings = [0. for x in range(pairwise_stats.num_players)]
        self.base = base
        self.spread = spread
        self.home_field_bias: float = home_field_bias
        self.draw_bias: float = draw_bias

    def update_ratings(
            self
    ):
        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            A: float = 0.0
            B: float = 0.0

            for opponent in range(self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result = self.pairwise_stats.statistics[player][opponent]

                if result.opponent_idx > player:
                    opponent_rating = self.next_ratings[result.opponent_idx]
                else:
                    opponent_rating = self.ratings[result.opponent_idx]

                A += result.w_ij + result.d_ij + result.l_ji + result.d_ji

                B += ((result.d_ij + result.w_ij) * self.home_field_bias /
                        (self.home_field_bias * self.ratings[player] +
                        self.draw_bias * opponent_rating) +
                        (result.d_ij + result.l_ij) * self.draw_bias * self.home_field_bias /
                        (self.draw_bias * self.home_field_bias * self.ratings[player] +
                        opponent_rating) +
                        (result.d_ji + result.w_ji) * self.draw_bias /
                        (self.home_field_bias * opponent_rating +
                        self.draw_bias * self.ratings[player]) +
                        (result.d_ji + result.l_ji) /
                        (self.draw_bias * self.home_field_bias * opponent_rating +
                        self.ratings[player]))

            self.next_ratings[player] = A / B

        self.ratings, self.next_ratings = self.next_ratings, self.ratings

    def update_home_field_bias(self):
        numerator = 0.
        denominator = 0.

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            for opponent in range(self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result = self.pairwise_stats.statistics[player][opponent]
                opponent_rating = self.ratings[result.opponent_idx]

                numerator += result.w_ij + result.d_ij
                denominator += ((result.d_ij + result.w_ij) * self.ratings[player] /
                                (self.home_field_bias * self.ratings[player] +
                                self.draw_bias * opponent_rating) +
                                (result.d_ij + result.l_ij) * self.draw_bias *
                                self.ratings[player] /
                                (self.draw_bias * self.home_field_bias *
                                self.ratings[player] + opponent_rating))

        return numerator / denominator

    def update_draw_bias(self):
        numerator = 0.
        denominator = 0.

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            for opponent in range(self.pairwise_stats.num_opponents_per_player[player]-1, -1, -1):
                result = self.pairwise_stats.statistics[player][opponent]
                opponent_rating = self.ratings[result.opponent_idx]

                numerator += result.d_ij
                denominator += ((result.d_ij + result.w_ij) * opponent_rating /
                                (self.home_field_bias * self.ratings[player] +
                                self.draw_bias * opponent_rating) +
                                (result.d_ij + result.l_ij) * self.home_field_bias *
                                self.ratings[player] /
                                (self.draw_bias * self.home_field_bias *
                                self.ratings[player] + opponent_rating))

        c = numerator / denominator
        return c + (c * c + 1)**0.5

    def compute_difference(
        self, n: int,
        pd1: "list[float]", pd2: "list[float]"
    ):
        result = 0.
        for i in range(n-1, -1, -1):
            diff = abs(pd1[i] - pd2[i]) / (pd1[i] + pd2[i])
            if diff > result:
                result = diff
        return result

    def minorize_maximize(
        self,
        use_home_field_bias: bool = False,
        home_field_bias: float = 1.,
        use_draw_bias: bool = False,
        draw_bias: float = 1.,
        epsilon: float = 1e-5,
        iterations: int = 10000
    ):
        """_summary_

        Args:
            use_home_field_bias (bool, optional): _description_. Defaults to False.
            home_field_bias (float, optional): _description_. Defaults to 1..
            use_draw_bias (bool, optional): _description_. Defaults to False.
            draw_bias (float, optional): _description_. Defaults to 1..
            epsilon (float, optional): _description_. Defaults to 1e-5.
            iterations (int, optional): _description_. Defaults to 10000.
        """

        # Set initial values
        self.home_field_bias = home_field_bias
        self.draw_bias = draw_bias
        self.ratings = [1. for p in range(self.pairwise_stats.num_players)]

        # Main MM loop
        for player in range(iterations):
            self.update_ratings()
            diff = self.compute_difference(
                self.pairwise_stats.num_players,
                self.ratings, self.next_ratings
            )

            if not use_home_field_bias:
                new_home_field_bias = self.update_home_field_bias()
                home_field_bias_diff = \
                    abs(self.home_field_bias - new_home_field_bias)
                if home_field_bias_diff > diff:
                    diff = home_field_bias_diff
                self.home_field_bias = new_home_field_bias

            if not use_draw_bias:
                new_draw_bias = self.update_draw_bias()
                draw_bias_diff = abs(self.draw_bias - new_draw_bias)
                if draw_bias_diff > diff:
                    diff = draw_bias_diff
                self.draw_bias = new_draw_bias

            if diff < epsilon:
                break

        # Convert back to Elos

        total = 0.
        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            tmp_base = self.elos[player].base
            tmp_spread = self.elos[player].spread
            self.elos[player] = EloRate(
                log(self.ratings[player], self.base) * self.spread,
                self.elos[player].std)
            self.elos[player].base = tmp_base
            self.elos[player].spread = tmp_spread
            total += self.elos[player].mu

        offset = -total / self.pairwise_stats.num_players

        for player in range(self.pairwise_stats.num_players-1, -1, -1):
            tmp_base = self.elos[player].base
            tmp_spread = self.elos[player].spread
            self.elos[player] = EloRate(
                self.elos[player].mu + offset,
                self.elos[player].std)
            self.elos[player].base = tmp_base
            self.elos[player].spread = tmp_spread

        if not use_home_field_bias:
            self.eloAdvantage = \
                log(self.home_field_bias, self.base) * self.spread
        if not use_draw_bias:
            self.eloDraw = log(self.draw_bias, self.base) * self.spread


def bayeselo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", elo_base: float = 10., elo_spread: float = 400.,
    elo_draw: float = 97.3, elo_advantage: float = 32.8,
) -> "list[EloRate]":

    if len(elos) != 0:
        base = elos[0].base
        spread = elos[0].spread

    for e in elos:
        if e.base != base or e.spread != spread:
            raise ValueError("Elos with different bases and \
                             spreads are not compatible")

    pairwise_stats = PopulationPairwiseStatistics.from_interactions(
        players=players,
        interactions=interactions
    )

    bt = BradleyTerryModel(
        pairwise_stats=pairwise_stats, elos=elos, elo_draw=elo_draw,
        elo_advantage=elo_advantage, base=base, spread=spread
    )

    use_home_field_bias = True
    use_draw_bias = True

    home_field_bias = base**(elo_advantage/spread)
    draw_bias = base**(elo_draw/spread)

    bt.minorize_maximize(
        use_home_field_bias=use_home_field_bias,
        use_draw_bias=use_draw_bias,
        home_field_bias=home_field_bias,
        draw_bias=draw_bias
    )

    # EloScale # TODO: Figure out what on earth that is
    for i, e in enumerate(bt.elos):
        x = e.base**(-elo_draw/e.spread)
        eloScale = x * 4.0 / ((1 + x) * (1 + x))
        tmp_base = bt.elos[i].base
        tmp_spread = bt.elos[i].spread
        bt.elos[i] = EloRate(
            bt.elos[i].mu * eloScale,
            bt.elos[i].std)
        bt.elos[i].base = tmp_base
        bt.elos[i].spread = tmp_spread

    return bt.elos
