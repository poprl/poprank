from math import log
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


def bayeselo(
     players: "list[str]", interactions: "list[Interaction]",
     elos: "list[EloRate]", eloDraw: float = 97.3, eloAdvantage: float = 32.8
     ) -> "list[EloRate]":

    class CCondensedResult:  # cr
        opponent_idx: int  # id of the opponent
        TrueGames: int  # True number of games played
        w_ij: float  # win player i against player j
        d_ij: float  # draw player i against player j
        l_ij: float  # loss player i against player j
        w_ji: float  # win player j against player i
        d_ji: float  # draw player j against player i
        l_ji: float  # loss player j against player i

        def __init__(self,
                     Opponent=-1,
                     TrueGames=0,
                     w_ij=0,
                     d_ij=0,
                     l_ij=0,
                     w_ji=0,
                     d_ji=0,
                     l_ji=0):
            self.opponent_idx = Opponent
            self.TrueGames = TrueGames
            self.w_ij = w_ij
            self.d_ij = d_ij
            self.l_ij = l_ij
            self.w_ji = w_ji
            self.d_ji = d_ji
            self.l_ji = l_ji

    class CCondensedResults:  # crs
        players: int  # Number of players in the pop
        p0pponents: "list[int]"  # Number of opponents for each player
        ppcr: "list[list[CCondensedResult]]"  # Results for each match

        def __init__(self, players, p0pponents, ppcr):
            self.players = players
            self.p0pponents = p0pponents
            self.ppcr = ppcr

        def getPlayers(self): return self.players
        def getOpponents(self, player: int): return self.p0pponents[player]

        def getCondensedResult(self, player: int, i: int):
            return self.ppcr[player][i]

    class CBradleyTerry:
        def __init__(self,
                     crs,
                     velo,
                     eloAdvantage=32.8,
                     eloDraw=97.3,
                     base=10,
                     spread=400,
                     ThetaW=0,
                     ThetaD=0):
            self.results: CCondensedResults = crs  # Condensed results
            self.velo = velo  # Players elos
            self.eloAdvantage = eloAdvantage  # advantage of playing white
            self.eloDraw = eloDraw  # likelihood of drawing
            self.v1 = crs.getPlayers()
            self.v2 = crs.getPlayers()
            self.ratings = [0. for x in range(self.v1)]
            self.next_ratings = [0. for x in range(self.v2)]
            self.base = base
            self.spread = spread
            self.home_field_bias: float = ThetaW
            self.draw_bias: float = ThetaD

        def update_ratings(self):
            for player in range(self.results.getPlayers()-1, -1, -1):
                A: float = 0
                B: float = 0

                for opponent in range(self.results.getOpponents(player)-1, -1, -1):
                    result = self.results.getCondensedResult(player, opponent)

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

            for player in range(self.results.getPlayers()-1, -1, -1):
                for opponent in range(self.results.getOpponents(player)-1, -1, -1):
                    result = self.results.getCondensedResult(player, opponent)
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

            for player in range(self.results.getPlayers()-1, -1, -1):
                for opponent in range(self.results.getOpponents(player)-1, -1, -1):
                    result = self.results.getCondensedResult(player, opponent)
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

        def GetDifference(self, n: int,
                          pd1: "list[float]", pd2: "list[float]"):
            result = 0.
            for i in range(n-1, -1, -1):
                diff = abs(pd1[i] - pd2[i]) / (pd1[i] + pd2[i])
                if diff > result:
                    result = diff
            return result

        def ConvertEloToGamma(self):
            self.home_field_bias = self.base**(self.eloAdvantage/self.spread)
            self.draw_bias = self.base**(self.eloDraw/self.spread)
            for i, e in enumerate(self.velo):
                self.ratings[i] = self.base**(e.mu/self.spread)

        def MinorizationMaximization(
                self,
                use_home_field_bias: bool = False,
                use_draw_bias: bool = False,
                home_field_bias: float = 1.,
                draw_bias: float = 1.,
                epsilon: float = 1e-5,
                iterations: int = 10000):

            # Set initial values
            self.home_field_bias = home_field_bias
            self.draw_bias = draw_bias
            self.ratings = [1. for p in range(self.results.getPlayers())]

            # Main MM loop
            for player in range(iterations):
                self.update_ratings()
                diff = self.GetDifference(self.results.getPlayers(),
                                          self.ratings, self.next_ratings)

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
            for player in range(self.results.getPlayers()-1, -1, -1):
                tmp_base = self.velo[player].base
                tmp_spread = self.velo[player].spread
                self.velo[player] = EloRate(
                    log(self.ratings[player], self.base) * self.spread,
                    self.velo[player].std)
                self.velo[player].base = tmp_base
                self.velo[player].spread = tmp_spread
                total += self.velo[player].mu

            offset = -total / self.results.getPlayers()

            for player in range(self.results.getPlayers()-1, -1, -1):
                tmp_base = self.velo[player].base
                tmp_spread = self.velo[player].spread
                self.velo[player] = EloRate(
                    self.velo[player].mu + offset,
                    self.velo[player].std)
                self.velo[player].base = tmp_base
                self.velo[player].spread = tmp_spread

            if not use_home_field_bias:
                self.eloAdvantage = \
                    log(self.home_field_bias, self.base) * self.spread
            if not use_draw_bias:
                self.eloDraw = log(self.draw_bias, self.base) * self.spread

    def countTrueGames(player, p0pponents, ppcr):
        result = 0
        for i in range(p0pponents[player]):
            result += ppcr[player][i].TrueGames
        return result

    def findOpponent(player, opponent, p0pponents, ppcr):
        for x in range(p0pponents[player]):
            if ppcr[player][x].opponent_idx == opponent:
                return ppcr[player][x]
        raise RuntimeError("Cound not find opponent")

    base = 10.
    spread = 400.

    if len(elos) != 0:
        base = elos[0].base
        spread = elos[0].spread

    for e in elos:
        if e.base != base or e.spread != spread:
            raise ValueError("Elos with different bases and \
                             spreads are not compatible")

    p0pponents = [0 for p in players]
    ppcr = [[] for p in players]
    ppcr_ids = [[] for p in players]
    indx = {p: i for i, p in enumerate(players)}

    for i in interactions:

        if i.players[1] not in\
          ppcr_ids[indx[i.players[0]]]:
            # If the players have never played together before

            # Add player 1 to the list of opponents of player 0
            ppcr_ids[indx[i.players[0]]].append(i.players[1])
            ppcr[indx[i.players[0]]].append(CCondensedResult(
                Opponent=indx[i.players[1]],
            ))
            p0pponents[indx[i.players[0]]] += 1

            # Add player 0 to the list of opponents of player 1
            ppcr_ids[indx[i.players[1]]].append(i.players[0])
            ppcr[indx[i.players[1]]].append(CCondensedResult(
                Opponent=indx[i.players[0]],
            ))
            p0pponents[indx[i.players[1]]] += 1

        if i.outcomes[0] > i.outcomes[1]:  # White wins
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].w_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].w_ji += 1

        elif i.outcomes[0] < i.outcomes[1]:  # Black wins
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].l_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].l_ji += 1

        else:  # Draw
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].d_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].d_ji += 1

        # Update true games of player 0
        tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
        ppcr[indx[i.players[0]]][tmp].TrueGames += 1

        # Update true games of player 1
        tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
        ppcr[indx[i.players[1]]][tmp].TrueGames += 1

    # addPrior
    priorDraw = 2.
    for p, cr in enumerate(ppcr):
        prior = priorDraw * 0.25 / countTrueGames(p, p0pponents, ppcr)
        for j in range(p0pponents[p]):
            crPlayer = ppcr[p][j]
            crOpponent = findOpponent(crPlayer.opponent_idx, p, p0pponents, ppcr)
            thisPrior = prior * crPlayer.TrueGames
            crPlayer.d_ij += thisPrior
            crPlayer.d_ji += thisPrior
            crOpponent.d_ij += thisPrior
            crOpponent.d_ji += thisPrior

    crs = CCondensedResults(players=len(players),
                            p0pponents=p0pponents,
                            ppcr=ppcr)

    bt = CBradleyTerry(crs, elos, eloDraw=eloDraw, eloAdvantage=eloAdvantage,
                       base=base, spread=spread)

    use_home_field_bias = True
    use_draw_bias = True

    home_field_bias = base**(eloAdvantage/spread)
    draw_bias = base**(eloDraw/spread)

    bt.MinorizationMaximization(use_home_field_bias=use_home_field_bias, use_draw_bias=use_draw_bias, home_field_bias=home_field_bias, draw_bias=draw_bias)

    # EloScale # TODO: Figure out what on earth that is
    for i, e in enumerate(bt.velo):
        x = e.base**(-eloDraw/e.spread)
        eloScale = x * 4.0 / ((1 + x) * (1 + x))
        tmp_base = bt.velo[i].base
        tmp_spread = bt.velo[i].spread
        bt.velo[i] = EloRate(
            bt.velo[i].mu * eloScale,
            bt.velo[i].std)
        bt.velo[i].base = tmp_base
        bt.velo[i].spread = tmp_spread

    return bt.velo
