from math import log
from popcore import Interaction, Player

from poprank import Rate
from poprank.functional.math import sigmoid
from poprank.utils import to_pairwise
from .wdl import windrawlose


class EloRate(Rate):
    """Elo rating.

    :param float base: base of the exponent in the elo formula
    :param float spread: divisor of the exponent in the elo formula

    See also :meth:`poprank.functional.elo.elo()`,
    :meth:`poprank.functional.bayeselo.bayeselo()`
    """
    # TODO: base and spread should be parameters.

    def __init__(
        self, mu: float = 0.0, std: float = 1.0,
        base: float = 10, spread: float = 400.0
    ):
        super().__init__(mu, std)
        self.base = base
        self.spread = spread

    def predict(self, opponent_elo: "EloRate") -> float:
        """Return the expected score against an opponent of the specified elo

        Uses the elo formula with self.base and self.spread substituted

        :parameter opponent_elo: (Rate) the elo of the opponent
        :return: The expected score.
        :rtype: float"""
        if not isinstance(opponent_elo, EloRate):
            raise TypeError("opponent_elo should be of type EloRate")

        return sigmoid(
            (opponent_elo.mu - self.mu) / self.spread, base=self.base)

    @property
    def q(self):
        return log(self.base) / self.spread

    def __repr__(self) -> str:
        return (
            f"EloRate(mu={self.mu}, std={self.std},"
            f"base={self.base}, spread={self.spread})"
        )


def _elo_update(
    elo: EloRate, true_score: float,
    expected_score: float, k_factor: float
) -> float:
    """
        Performs the Elo update
    """
    return elo.mu + k_factor * (true_score - expected_score)


def _agg(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float, wdl: bool
):
    """_summary_

    :param population: _description_
    :type population: Population
    :param interactions: _description_
    :type interactions: list[Interaction]
    :param elos: _description_
    :type elos: list[EloRate]
    :param k_factor: _description_
    :type k_factor: float
    :param wdl: _description_
    :type wdl: bool
    :return: _description_
    :rtype: _type_
    """

    exp_scores = [.0 for _ in players]
    true_scores = [.0 for _ in players]

    for interaction in interactions:
        player = players.index(interaction.players[0])
        opponent = players.index(interaction.players[1])

        exp_scores[player] += elos[player].predict(elos[opponent])
        exp_scores[opponent] += elos[opponent].predict(elos[player])

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
    u_elos: "list[EloRate]" = []
    for idx, elo in enumerate(elos):
        u_elo = _elo_update(
            elo, true_score=true_scores[idx],
            expected_score=exp_scores[idx], k_factor=k_factor
        )
        u_elos.append(EloRate(u_elo, elo.std))

    return u_elos


def _stream(
    players: "list[Player]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float, wdl: bool
):
    """
        TODO:
    """
    u_elos = [EloRate(o_elo.mu, o_elo.std) for o_elo in elos]

    for interaction in interactions:
        player = players.index(interaction.players[0])
        opponent = players.index(interaction.players[1])

        u_elos[player].mu = _elo_update(
            elo=u_elos[player], true_score=interaction.outcomes[0],
            expected_score=u_elos[player].predict(u_elos[opponent]),
            k_factor=k_factor
        )
        elos[opponent].mu = _elo_update(
            elo=u_elos[opponent], true_score=interaction.outcomes[1],
            expected_score=u_elos[opponent].predict(u_elos[player]),
            k_factor=k_factor
        )

    return u_elos


def elo(
    players: "list[Player]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float = 20,
    wdl: bool = False, reduce: str = "aggregate"
) -> "list[EloRate]":
    """Rates players by calculating their new elo after a set of interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5).

    It is important to note that applying elo to a set of interactions is not
    equivalent to applying elo to each interaction sequentially. To reduce the
    set by aggregating all interaction use `reduce`= aggregate (default). To 
    compute ratings after each interaction, use `reduce` = stream.

    :param list[str] players: A list containing all unique player identifiers
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly 2
        players and result in a win (1, 0), a loss (0, 1)
        or a draw (0.5, 0.5)
    :param list[EloRate] elos: The initial ratings of the players
    :param float k_factor: Maximum possible adjustment per game. Larger means
        player rankings change faster
    :param str reduce: The aggregation method used to reduce the interactions. Values
        can be either "aggregate" or "stream".
    :param bool wdl: Turn the interactions into the (1, 0), (.5, .5),
        (0, 1) format automatically. Defaults to False.

    :raises ValueError: If the numbers of players and ratings don't match,
            If an interaction has the wrong number of players,
            If an interaction has the wrong number of outcomes,
            If a player that does not appear in `players`is in an interaction
    :raises TypeError: Using Rate instead of EloRate

    :return: The updated ratings of all players
    :rtype: list[EloRate]

    Example
    -------

    It is important to note that applying elo to a set of interactions is not
    equivalent to applying elo to each interaction sequentially.

    .. code-block:: python

        # Example applying elo to all interactions at once
        # with aggregated reduction

        from poprank.functional.elo import elo
        from poprank import EloRate
        from popcore import Interaction


        players=["a", "b", "c", "d", "e", "f"]
        interactions=[
            Interaction(["a", "b"], [0, 1]),
            Interaction(["a", "c"], [0.5, 0.5]),
            Interaction(["a", "d"], [1, 0]),
            Interaction(["a", "e"], [1, 0]),
            Interaction(["a", "f"], [0, 1])
        ]
        elos=[
            EloRate(1613, 0),
            EloRate(1609, 0),
            EloRate(1477, 0),
            EloRate(1388, 0),
            EloRate(1586, 0),
            EloRate(1720, 0)
        ]
        k_factor=32

        new_ratings = elo(players, interactions, elos, k_factor, wdl)

        # new_ratings is equal to
        # [EloRate(1601, 0), EloRate(1625, 0), EloRate(1483, 0),
        # EloRate(1381, 0), EloRate(1571, 0), EloRate(1731, 0)]

    .. code-block:: python

        # Example applying elo to all interactions sequentially

        from poprank.functional.elo import elo
        from poprank import EloRate
        from popcore import Interaction

        players=["a", "b", "c", "d", "e", "f"]
        interactions=[
            Interaction(["a", "b"], [0, 1]),
            Interaction(["a", "c"], [0.5, 0.5]),
            Interaction(["a", "d"], [1, 0]),
            Interaction(["a", "e"], [1, 0]),
            Interaction(["a", "f"], [0, 1])
        ]
        elos=[
            EloRate(1613, 0),
            EloRate(1609, 0),
            EloRate(1477, 0),
            EloRate(1388, 0),
            EloRate(1586, 0),
            EloRate(1720, 0)
        ]
        k_factor=32

        for i in interactions:
            elos = elo(players, [i], elos, k_factor)

        # elos is equal to
        # [EloRate(1603.1911843424746, 0)
        # EloRate(1625.1841986691566, 0),
        # EloRate(1482.308914159826, 0),
        # EloRate(1380.429196623972, 0),
        # EloRate(1570.601965641378, 0),
        # EloRate(1731.2845405631929, 0)]

    .. seealso::
        :meth:`poprank.functional.bayeselo`

        :class:`poprank.rates.EloRate`
    """

    # Checks
    if len(players) != len(elos):
        raise ValueError("Players and elos length mismatch"
                         f": {len(players)} != {len(elos)}")

    for interaction in interactions:
        if not wdl and (interaction.outcomes[0] not in (0, .5, 1) or
                        interaction.outcomes[1] not in (0, .5, 1) or
                        sum(interaction.outcomes) != 1):
            raise Warning("Elo takes outcomes in the (1, 0), (0, 1), (.5, .5) "
                          "format, other values may have unspecified behavior "
                          "(set wdl=True to automatically turn interactions "
                          "into the windrawlose format)")

    # Calculate the expected score vs true score of all players in the given
    # set of interactions and adjust elo afterwards accordingly.

    interactions = to_pairwise(interactions)

    if reduce == "aggregate":
        rates = _agg(players, interactions, elos, k_factor, wdl)
    elif reduce == "stream":
        rates = _stream(players, interactions, elos, k_factor, wdl)
    else:
        raise ValueError("reduce")

    return rates
