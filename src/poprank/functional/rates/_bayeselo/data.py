from dataclasses import dataclass
from popcore import Interaction


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


@dataclass
class BayesEloStats:  # crs
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
            prior: float = draw_prior * 0.25 / self.count_total_opponent_games(
                player)

            for opponent in range(self.num_opponents_per_player[player]):
                cr_player = self.statistics[player][opponent]
                cr_opponent = self.find_opponent(
                    cr_player.opponent_idx, player)
                this_prior: float = prior * cr_player.total_games
                cr_player.d_ij += this_prior
                cr_player.d_ji += this_prior
                cr_opponent.d_ij += this_prior
                cr_opponent.d_ji += this_prior

    def find_opponent(
        self,
        player_idx: int,
        opponent_idx: int,
    ) -> PairwiseStatistics:
        """Return the pairwise interaction statistics between the player
        and the opponent

        Args:
            player_idx (int): Id of the player
            opponent_idx (int): Id of the opponent

        Raises:
            RuntimeError: If the opponent could not be foud
        """
        for x in range(self.num_opponents_per_player[player_idx]):
            if self.statistics[player_idx][x].opponent_idx == opponent_idx:
                return self.statistics[player_idx][x]
        raise RuntimeError(f"Cound not find opponent {opponent_idx} \
                        for player {player_idx}")

    def count_total_opponent_games(
        self,
        player_idx: int,
    ) -> int:
        """Return the sum of all games played by opponents of the player

        Args:
            player_idx (int): Id of the player
        """

        return sum([
            opponent.total_games for opponent in self.statistics[player_idx]
        ])

    @staticmethod
    def from_interactions(
        players: 'list[str]',
        interactions: 'list[Interaction]',
        add_draw_prior: bool = True,
        draw_prior: float = 2.0
    ) -> 'BayesEloStats':
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

        pps: BayesEloStats = BayesEloStats(
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
