import numpy as np

from popcore import Interaction
from poprank.functional.rates import MultidimEloRate
from poprank.functional.rates.melo import _build_omega, _melo_predict


def bipartite_multidim_elo(
    players: "list[str]",
    interactions: "list[Interaction]",
    player_elos: "list[MultidimEloRate]",
    opponents: "list[str]" = None,
    opponents_elos: "list[MultidimEloRate]" = None,
    k: int = 1, lr1: float = 16, lr2: float = 1, iterations: int = 100
) -> "tuple[list[MultidimEloRate]]":
    """Computes the multidimensional elo ratings of the players based on the
    interactions against opponents rather than between each other.

    This method of rating is non-transitive.
    Based on https://arxiv.org/abs/1806.02643.

    :param list[str] players: A list containing all unique player identifiers.
    :param list[str] tasks:  A list containing all unique task identifiers.
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly one
        player and one task (in this order) and have outcomes in the format
        [p, 1-p] where 0<=p<=1.
    :param list[MeloRate] player_elos: The initial ratings of the players.
        Must have the same k as what's passed as argument in this method.
    :param list[MeloRate] player_elos: The initial ratings of the tasks.
        Must have the same k as what's passed as argument in this method.
    :param Optional[int] k: Use mElo with 2k dimensions. Must be the same k as
        in the elos. Defaults to 1.
    :param Optional[float] lr1: Learning rate of the ratings. Defaults to 16.
    :param Optional[float] lr2: Learning rate of the vectors. Defaults to 1.

    :returns: Two lists, the first one of the updated player ratings and the
        second of the updated task ratings.
    :rtype: tuple[list[MeloRate]]

    Example
    -------

    .. code-block:: python

        from poprank.functional.melo import mEloAvT
        from poprank import MeloRate
        from popcore import Interaction

        k = 1
        players = ["a", "b", "c"]
        tasks = ["d", "e"]
        interac = []

        for i in range(100):    # Needs enough cases to converge
            interac.extend([
                Interaction(["a", "d"], [1, 0]),
                Interaction(["b", "d"], [0, 1]),
                Interaction(["c", "d"], [1, 0]),
                Interaction(["a", "e"], [1, 0]),
                Interaction(["b", "e"], [0, 1]),
                Interaction(["c", "e"], [1, 0]),
            ])

        shuffle(interac)

        player_elos = [MeloRate(0, 1, k=k) for p in players]
        task_elos = [MeloRate(0, 1, k=k) for t in tasks]

        player_elos, task_elos = mEloAvT(
            players, tasks, interac, player_elos, task_elos,
            k=k, lr1=1, lr2=0.1)

        # Display the expected outcomes of the matches between players
        # Format is (correct answer, mElo expected outcome)

        print(1., round(player_elos[0].expected_outcome(task_elos[0]), 3))
        print(0., round(player_elos[1].expected_outcome(task_elos[0]), 3))
        print(1., round(player_elos[2].expected_outcome(task_elos[0]), 3))
        print(1., round(player_elos[0].expected_outcome(task_elos[1]), 3))
        print(0., round(player_elos[1].expected_outcome(task_elos[1]), 3))
        print(1., round(player_elos[2].expected_outcome(task_elos[1]), 3))

    .. seealso::
        :class:`poprank.rates.MeloRate`

        :meth:`poprank.functional.mElo`

        :meth:`poprank.functional.elo`
    """

    # new_player_elos = deepcopy(player_elos)
    # new_task_elos = deepcopy(opponents_elos)

    players_rates = np.array([e.mu for e in player_elos])
    opponents_rates = np.array(
        [e.mu for e in opponents_elos]) if opponents else players_rates

    # Initialize U and V matrices
    p_cyclic = np.array([e.cyclic for e in player_elos])
    o_cyclic = np.array(
        [e.cyclic for e in opponents_elos]) if opponents else p_cyclic

    omega = _build_omega(k)

    players_idx = {p: idx for idx, p in enumerate(players)}
    opponents_idx = {
        o: idx for idx, o in enumerate(opponents)
    } if opponents else players_idx

    for i in range(iterations):
        np.random.shuffle(interactions)
        for interac in interactions:
            player = players_idx[interac.players[0]]
            opponent = opponents_idx[interac.players[1]]

            # Expected win probability
            expected_outcome = _melo_predict(
                players_rates[player], p_cyclic[player],
                opponents_rates[opponent], o_cyclic[opponent],
                omega
            )
            # Delta between expected and actual win
            # I had to change the index here, and I don't know why
            # I am so confused
            delta = interac.outcomes[0] - expected_outcome

            # Update ratings. r has higher lr than c
            players_rates[player] += lr1*delta
            opponents_rates[opponent] += -lr1*delta

            cyclic_player = lr2 * delta * (omega @ o_cyclic[opponent]).T
            cyclic_oppon = -lr2 * delta * (omega @ p_cyclic[player]).T

            p_cyclic[player] += cyclic_player
            o_cyclic[opponent] += cyclic_oppon

    players = [
        MultidimEloRate(r, k=k, cyclic=c)
        for r, c in zip(players_rates, iter(p_cyclic))
    ]

    if opponents:
        opponents = [
            MultidimEloRate(r, k=k, cyclic=c)
            for r, c in zip(opponents_rates, iter(o_cyclic))
        ]
        return players, opponents

    return players
