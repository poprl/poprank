
from typing import Optional
import numpy as np

from popcore import Interaction

from ..math import sigmoid
from ..._core import Rate


def _melo_predict(
    rate1: float, cyclic1: np.ndarray,
    rate2: float, cyclic2: np.ndarray,
    omega: np.ndarray
) -> float:
    """
       Computes the mElo win probability of a player against an opponent based
       on their rating and cyclic component approximations.

    :param rate1: Rate of the first player.
    :type rate1: float
    :param cyclic1: Cyclic component of the first player.
    :type cyclic1: np.ndarray
    :param rate2: Rate of the second player
    :type rate2: float
    :param cyclic2: Cyclic component of the second player.
    :type cyclic2: np.ndarray
    :param omega: A 2k x 2k matrix with off-diagonal +1 and -1 elements.
        See _build_omega.
    :type omega: np.ndarray
    :return: Winning probability of the player.
    :rtype: float
    """
    two_k = len(cyclic1)
    assert two_k == len(cyclic2)
    assert omega.shape == (two_k, two_k)

    return sigmoid(
        rate1 - rate2 + cyclic1 @ omega @ cyclic2
    )


def _build_omega(k: int) -> np.ndarray:
    """
        Constructs a 2k x 2k matrix with alternating off-diagonal
        +1 and -1 elements.

    :param k: mElo2k order.
    :type k: int
    :return: a 2k x 2k matrix with alternating off-diagonal
        +1 and -1 elements.
    :rtype: np.ndarray
    """
    omega = np.zeros([2 * k, 2 * k])
    idx = 2 * np.arange(k)
    omega[idx, idx + 1] = 1.0
    omega[idx + 1, idx] = -1.0
    return omega


class MultidimEloRate(Rate):
    """mElo2k rating.

    :param float mu: Player's initial rating. Defaults to 0.
    :param float std: Player's default standard deviation. Defaults to 1
    :param int k: The mElo rating will have 2k dimensions. Defaults to 1.
    :param list vector: The initial mElo vector. Should be of length 2k. If
        None, it will be initialized to a uniform[-0.5, 0.5] random vector of
        length 2k. Defaults to None.
    """
    # TODO: Test behavior for k = 0
    def __init__(self, mu: float, std: float = 1.0, k: int = 1,
                 cyclic: Optional[np.ndarray] = None):
        super().__init__(mu, std)

        self.k = k
        self.omega = _build_omega(self.k)

        if cyclic is None:
            # NOTE: no seeding required at this level
            # to guarantee consistency, set the seed at numpy level.
            cyclic = np.random.uniform(-0.5, 0.5, size=2*k)
        assert len(cyclic) == 2 * k, "The vector must be of length 2k"
        self.cyclic = cyclic

    def predict(self, other: "MultidimEloRate") -> float:
        """Expected score of the player against an opponent with the specified
        rating.

        :param other: the multidimensional Elo rate of the other player.
        :type other: MultidimEloRate
        :return: expected score.
        :rtype: float
        """
        assert other.k == self.k  # TODO: exception raising

        return _melo_predict(
            self.mu, self.cyclic, other.mu, other.cyclic, self.omega)

    def __repr__(self) -> str:
        return f"MultidimEloRate(mu={self.mu}, std={self.std}, cyc={str(self.cyclic)})"  # noqa


def multidim_elo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[MultidimEloRate]", k: int = 1, lr1: float = 16, lr2: float = 1,
    iterations: Optional[int] = 100
) -> "list[MultidimEloRate]":
    """Computes the multidimensional elo ratings of the players based on the
    interactions.

    This method of rating capture non-transitive relationships.
    Introduced on https://arxiv.org/abs/1806.02643.

    :param list[str] players: A list containing all unique player identifiers.
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly 2
        players and have outcomes in the format [p, 1-p] where 0<=p<=1.
    :param list[MeloRate] elos: The initial ratings of the players. Must have
        the same k as what's passed as argument in this method.
    :param Optional[int] k: Use mElo with 2k dimensions. Must be the same k as
        in the elos. Defaults to 1.
    :param Optional[float] lr1: Learning rate of the ratings. Defaults to 16.
    :param Optional[float] lr2: Learning rate of the cyclic components.
        Defaults to 1.
    :param Optional[int] iterations: number of iterations to perform the
        gradient descent updates. Defaults to 100, increase for accuracy.

    :returns: A list of the updated ratings.
    :rtype: list[MultidimEloRate]

    Example
    -------

    .. code-block:: python

        # Example using rock-paper-scissor

        from poprank.functional.melo import mElo
        from poprank import MultidimEloRate
        from popcore import Interaction

        k = 1
        players = ["a", "b", "c"]
        interactions = []

        for i in range(100):    # Needs enough cases to converge
            interactions.extend([
                Interaction(["a", "b"], [1, 0]),
                Interaction(["b", "c"], [1, 0]),
                Interaction(["c", "a"], [1, 0])
            ])

        elos = [MultidimEloRate(0, 1, k=k) for p in players]

        new_elos = multidim_elo(players, interactions, elos, k=k, lr1=1, lr2=0.1)

        # Display the expected outcomes of the matches between players
        # Format is (correct answer, mElo expected outcome)

        print(.5, round(new_elos[0].expected_outcome(new_elos[0]), 3))
        print(1., round(new_elos[0].expected_outcome(new_elos[1]), 3))
        print(0., round(new_elos[0].expected_outcome(new_elos[2]), 3))
        print(0., round(new_elos[1].expected_outcome(new_elos[0]), 3))
        print(.5, round(new_elos[1].expected_outcome(new_elos[1]), 3))
        print(1., round(new_elos[1].expected_outcome(new_elos[2]), 3))
        print(1., round(new_elos[2].expected_outcome(new_elos[0]), 3))
        print(0., round(new_elos[2].expected_outcome(new_elos[1]), 3))
        print(.5, round(new_elos[2].expected_outcome(new_elos[2]), 3))


    .. seealso::
        :class:`poprank.rates.MeloRate`

        :meth:`poprank.functional.mEloAvT`

        :meth:`poprank.functional.elo`
    """
    # return bipartite_multidim_elo(
    #     players, interactions, elos, k=k, lr1=lr1,
    #     lr2=lr2, iterations=iterations
    # )
    assert len(players) == len(elos), "Elos and players must match"
    assert all(e.k == k for e in elos), "K value missmatch"

    rates = np.array([rate.mu for rate in elos], dtype=np.float32)
    cyclic = np.array([rate.cyclic for rate in elos], dtype=np.float32)
    omega = _build_omega(k)

    player_indices = {p: i for i, p in enumerate(players)}
    for i in range(iterations):
        np.random.shuffle(interactions)
        for interaction in interactions:
            player = player_indices[interaction.players[0]]
            opponent = player_indices[interaction.players[1]]
            player_outcome, opponent_outcome = interaction.outcomes

            expected_outcome = _melo_predict(
                rates[player], cyclic[player],
                rates[opponent], cyclic[opponent],
                omega
            )

            delta = player_outcome - expected_outcome

            # print(interaction, delta)

            rates[player] += lr1 * delta
            rates[opponent] += -lr1 * delta

            o_c = np.matmul(omega, cyclic[[player, opponent]].T)
            cyclic_player = lr2 * delta * o_c[:, 1]
            cyclic_oppon = -lr2 * delta * o_c[:, 0]

            cyclic[player] += cyclic_player
            cyclic[opponent] += cyclic_oppon

    return [
        MultidimEloRate(r, k=k, cyclic=c)
        for r, c in zip(rates, iter(cyclic))
    ]


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
