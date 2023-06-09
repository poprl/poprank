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
        Opponent: int  # id of the opponent
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
            self.Opponent = Opponent
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
            self.crs = crs  # Condensed results
            self.velo = velo  # Players elos
            self.eloAdvantage = eloAdvantage  # advantage of playing white
            self.eloDraw = eloDraw  # likelihood of drawing
            self.v1 = crs.getPlayers()
            self.v2 = crs.getPlayers()
            self.pGamma = [0. for x in range(self.v1)]
            self.pNextGamma = [0. for x in range(self.v2)]
            self.base = base
            self.spread = spread
            self.ThetaW: float = ThetaW
            self.ThetaD: float = ThetaD

        def updateGammas(self):
            for player in range(self.crs.getPlayers()-1, -1, -1):
                A: float = 0
                B: float = 0

                for j in range(self.crs.getOpponents(player)-1, -1, -1):
                    cr = self.crs.getCondensedResult(player, j)

                    if cr.Opponent > player:
                        OpponentGamma = self.pNextGamma[cr.Opponent]
                    else:
                        OpponentGamma = self.pGamma[cr.Opponent]

                    A += cr.w_ij + cr.d_ij + cr.l_ji + cr.d_ji

                    B += ((cr.d_ij + cr.w_ij) * self.ThetaW /
                          (self.ThetaW * self.pGamma[player] +
                          self.ThetaD * OpponentGamma) +
                          (cr.d_ij + cr.l_ij) * self.ThetaD * self.ThetaW /
                          (self.ThetaD * self.ThetaW * self.pGamma[player] +
                          OpponentGamma) +
                          (cr.d_ji + cr.w_ji) * self.ThetaD /
                          (self.ThetaW * OpponentGamma +
                          self.ThetaD * self.pGamma[player]) +
                          (cr.d_ji + cr.l_ji) /
                          (self.ThetaD * self.ThetaW * OpponentGamma +
                          self.pGamma[player]))

                self.pNextGamma[player] = A / B

            self.pGamma, self.pNextGamma = self.pNextGamma, self.pGamma

        def UpdateThetaW(self):
            numerator = 0.
            denominator = 0.

            for player in range(self.crs.getPlayers()-1, -1, -1):
                for j in range(self.crs.getOpponents(player)-1, -1, -1):
                    cr = self.crs.getCondensedResult(player, j)
                    opponentGamma = self.pGamma[cr.Opponent]

                    numerator += cr.w_ij + cr.d_ij
                    denominator += ((cr.d_ij + cr.w_ij) * self.pGamma[player] /
                                    (self.ThetaW * self.pGamma[player] +
                                    self.ThetaD * opponentGamma) +
                                    (cr.d_ij + cr.l_ij) * self.ThetaD *
                                    self.pGamma[player] /
                                    (self.ThetaD * self.ThetaW *
                                    self.pGamma[player] + opponentGamma))

            return numerator / denominator

        def UpdateThetaD(self):
            numerator = 0.
            denominator = 0.

            for player in range(self.crs.getPlayers()-1, -1, -1):
                for j in range(self.crs.getOpponents(player)-1, -1, -1):
                    cr = self.crs.getCondensedResult(player, j)
                    opponentGamma = self.pGamma[cr.Opponent]

                    numerator += cr.d_ij
                    denominator += ((cr.d_ij + cr.w_ij) * opponentGamma /
                                    (self.ThetaW * self.pGamma[player] +
                                    self.ThetaD * opponentGamma) +
                                    (cr.d_ij + cr.l_ij) * self.ThetaW *
                                    self.pGamma[player] /
                                    (self.ThetaD * self.ThetaW *
                                    self.pGamma[player] + opponentGamma))

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
            self.ThetaW = self.base**(self.eloAdvantage/self.spread)
            self.ThetaD = self.base**(self.eloDraw/self.spread)
            for i, e in enumerate(self.velo):
                self.pGamma[i] = self.base**(e.mu/self.spread)

        def MinorizationMaximization(self, fThetaW: int,
                                     fThetaD: int, Epsilon: float = 1e-5):

            # Set initial values
            self.ThetaW = 1.0 if fThetaW else\
                self.base**(self.eloAdvantage/self.spread)
            self.ThetaD = 1.0 if fThetaD else\
                self.base**(self.eloDraw/self.spread)
            self.pGamma = [1. for p in range(self.crs.getPlayers())]

            # Main MM loop
            for i in range(10000):
                self.updateGammas()
                diff = self.GetDifference(self.crs.getPlayers(),
                                          self.pGamma, self.pNextGamma)

                if fThetaW:
                    newThetaW = self.UpdateThetaW()
                    ThetaW_diff = abs(self.ThetaW - newThetaW)
                    if ThetaW_diff > diff:
                        diff = ThetaW_diff
                    self.ThetaW = newThetaW

                if fThetaD:
                    newThetaD = self.UpdateThetaD()
                    ThetaD_diff = abs(self.ThetaD - newThetaD)
                    if ThetaD_diff > diff:
                        diff = ThetaD_diff
                    self.ThetaD = newThetaD

                if diff < Epsilon:
                    break

                if (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}: {diff}")

            # Convert back to Elos

            total = 0.
            for i in range(self.crs.getPlayers()-1, -1, -1):
                tmp_base = self.velo[i].base
                tmp_spread = self.velo[i].spread
                self.velo[i] = EloRate(
                    log(self.pGamma[i], self.base) * self.spread,
                    self.velo[i].std)
                self.velo[i].base = tmp_base
                self.velo[i].spread = tmp_spread
                total += self.velo[i].mu

            offset = -total / self.crs.getPlayers()

            for i in range(self.crs.getPlayers()-1, -1, -1):
                tmp_base = self.velo[i].base
                tmp_spread = self.velo[i].spread
                self.velo[i] = EloRate(
                    self.velo[i].mu + offset,
                    self.velo[i].std)
                self.velo[i].base = tmp_base
                self.velo[i].spread = tmp_spread

            if fThetaW:
                self.eloAdvantage = log(self.ThetaW, self.base) * self.spread
            if fThetaD:
                self.eloDraw = log(self.ThetaD, self.base) * self.spread

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
                TrueGames=1,
                d_ij=1,
                d_ji=1
            ))
            p0pponents[indx[i.players[0]]] += 1

            # Add player 0 to the list of opponents of player 1
            ppcr_ids[indx[i.players[1]]].append(i.players[0])
            ppcr[indx[i.players[1]]].append(CCondensedResult(
                Opponent=indx[i.players[0]],
                TrueGames=1,
                d_ij=1,
                d_ji=1
            ))
            p0pponents[indx[i.players[1]]] += 1

        if i.outcomes == (1, 0):  # White wins
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].w_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].w_ji += 1

        elif i.outcomes == (0, 1):  # Black wins
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].l_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].l_ji += 1

        elif i.outcomes == (.5, .5):  # Draw
            # Update score of player 0
            tmp = ppcr_ids[indx[i.players[0]]].index(i.players[1])
            ppcr[indx[i.players[0]]][tmp].d_ij += 1

            # Update score of player 1
            tmp = ppcr_ids[indx[i.players[1]]].index(i.players[0])
            ppcr[indx[i.players[1]]][tmp].d_ji += 1

    crs = CCondensedResults(players=len(players),
                            p0pponents=p0pponents,
                            ppcr=ppcr)

    bt = CBradleyTerry(crs, elos, eloDraw=eloDraw, eloAdvantage=eloAdvantage,
                       base=base, spread=spread)

    fThetaW = 0
    fThetaD = 0

    bt.MinorizationMaximization(fThetaW=fThetaW, fThetaD=fThetaD)

    # EloScale # TODO: Figure out what on earth that is
    for i, e in enumerate(bt.velo):
        x = e.base**(-eloDraw/e.spread)
        tmp_base = bt.velo[i].base
        tmp_spread = bt.velo[i].spread
        bt.velo[i] = EloRate(
            bt.velo[i].mu * x * 4.0 / ((1 + x) * (1 + x)),
            bt.velo[i].std)
        bt.velo[i].base = tmp_base
        bt.velo[i].spread = tmp_spread

    return bt.velo
