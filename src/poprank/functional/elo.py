from popcore import Interaction
from poprank import Rate, EloRate
from poprank.functional.wdl import windrawlose


def elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[EloRate]", k_factor: float = 20, wdl: bool = False
) -> "list[EloRate]":
    """Rates players by calculating their new elo after a set of interactions.

    Works for 2 players interactions, where each interaction can be
    a win (1, 0), a loss (0, 1) or a draw (0.5, 0.5).

    It is important to note that applying elo to a set of interactions is not
    equivalent to applying elo to each interaction sequentially.

    :param list[str] players: A list containing all unique player identifiers
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly 2
        players and result in a win (1, 0), a loss (0, 1)
        or a draw (0.5, 0.5)
    :param list[EloRate] elos: The initial ratings of the players
    :param float k_factor: Maximum possible adjustment per game. Larger means
        player rankings change faster
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

    for elo in elos:
        if not isinstance(elo, EloRate):
            raise TypeError("elos must be of type list[EloRate]")

    for interaction in interactions:
        if len(interaction.players) != 2 or len(interaction.outcomes) != 2:
            raise ValueError("Elo only accepts interactions involving "
                             "both a pair of players and a pair of outcomes")

        if interaction.players[0] not in players \
           or interaction.players[1] not in players:
            raise ValueError("Players(s) in interactions absent from player "
                             "list")

        if not wdl and (interaction.outcomes[0] not in (0, .5, 1) or
                        interaction.outcomes[1] not in (0, .5, 1) or
                        sum(interaction.outcomes) != 1):
            raise Warning("Elo takes outcomes in the (1, 0), (0, 1), (.5, .5) "
                          "format, other values may have unspecified behavior "
                          "(set wdl=True to automatically turn interactions "
                          "into the windrawlose format)")

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
        [EloRate(e.mu + k_factor*(true_scores[i] - expected_scores[i]), e.std)
         for i, e in enumerate(elos)]

    return rates
