from math import e
import random
from typing import Optional
import numpy as np
from copy import deepcopy

from popcore import Interaction

from ..math import sigmoid
from ..core import Rate


class MeloRate(Rate):
    """mElo2k rating.

    :param float mu: Player's initial rating. Defaults to 0.
    :param float std: Player's default standard deviation. Defaults to 1
    :param int k: The mElo rating will have 2k dimensions. Defaults to 1.
    :param list vector: The initial mElo vector. Should be of length 2k. If
        None, it will be initialized to a uniform[-0.5, 0.5] random vector of
        length 2k. Defaults to None.
    """
    # TODO: Test behavior for k = 0
    def __init__(self, mu: float, std: float, k: int = 1,
                 vector: Optional[list] = None):
        Rate.__init__(self, mu, std)
        if vector is None:
            # TODO: uncontrolled random behaviour!
            self.vector = [random.random() - 0.5 for x in range(2*k)]
        else:
            assert len(vector) == 2 * k, "The vector must be of length 2k"
            self.vector = vector
        self.k = k

    def _build_omega(self, k):
        omega = [[0 for x in range(2*k)] for y in range(2*k)]
        for i in range(k):
            omega[2*i][2*i+1] = 1
            omega[2*i+1][2*i] = -1
        return omega

    def predict(self, opponent: "MeloRate") -> float:
        """Expected score of the player against an opponent with the specified
        rating.

        :param MeloRate opponent: mElo2k Rate of the opponent. K must be the
            same for both players.

        :return: The expected score.
        :rtype: float
        """
        omega = self._build_omega(self.k)
        adjustment = [sum([i * j for i, j in zip(self.vector, omega[a])]) for a in range(self.k*2)]
        adjustment = sum([i*j for i, j in zip(adjustment, opponent.vector)])
        return sigmoid(self.mu - opponent.mu + adjustment, base=e, spread=1.0)


def _build_omega(k):
    omega = np.zeros((2*k, 2*k))
    e = np.atleast_3d(np.identity(2*k))
    for i in range(k):
        omega += e[:, 2*i] @ e[:, 2*i+1].T - e[2*i+1] @ e[2*i].T
    return omega


def mElo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[MeloRate]", k: int = 1, lr1: float = 16, lr2: float = 1
) -> "list[MeloRate]":
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
    :param Optional[float] lr2: Learning rate of the vectors. Defaults to 1.

    :returns: A list of the updated ratings.
    :rtype: list[MeloRate]

    Example
    -------

    .. code-block:: python

        # Example using rock-paper-scissor

        from poprank.functional.melo import mElo
        from poprank import MeloRate
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

        elos = [MeloRate(0, 1, k=k) for p in players]

        new_elos = mElo(players, interactions, elos, k=k, lr1=1, lr2=0.1)

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

    new_elos = deepcopy(elos)

    assert len(players) == len(new_elos), "Elos and players must match"
    for e in new_elos:
        assert e.k == k, "K value missmatch"

    # Outcomes must be in interval [0, 1]
    # k_factor must be positive int

    # Perhaps decompose an observed WIN/LOSS matrix in to a C Omega C'
    # for better initial params?

    # Initialize C matrix
    c_matrix = np.array([e.vector for e in elos])

    omega = _build_omega(k)

    player_indices = {p: i for i, p in enumerate(players)}

    for interaction in interactions:
        player = player_indices[interaction.players[0]]
        opponent = player_indices[interaction.players[1]]
        player_rating = new_elos[player].mu
        opponent_rating = new_elos[opponent].mu

        adjustment_matrix = c_matrix @ omega @ c_matrix.T
        player_adjust = adjustment_matrix[player, opponent]

        # Expected win probability
        win_prob = sigmoid(player_rating - opponent_rating + player_adjust)

        # Delta between expected and actual win
        delta = interaction.outcomes[0] - win_prob

        # Update ratings. r has higher lr than c
        new_elos[player].mu += lr1*delta
        new_elos[opponent].mu -= lr1*delta

        tmp_c_mat = np.array(c_matrix)

        tmp_c_mat[player] = \
            c_matrix[player] + lr2 * delta * (omega @ c_matrix[opponent]).T
        tmp_c_mat[opponent] = \
            c_matrix[opponent] - lr2 * delta * (omega @ c_matrix[player]).T

        c_matrix = tmp_c_mat

        new_elos[player].vector = list(c_matrix[player])
        new_elos[opponent].vector = list(c_matrix[opponent])

    return new_elos


def mEloAvT(
    players: "list[str]", tasks: "list[str]",
    interactions: "list[Interaction]",
    player_elos: "list[MeloRate]", task_elos: "list[MeloRate]",
    k: int = 1, lr1: float = 16, lr2: float = 1
) -> "tuple[list[MeloRate]]":
    """Computes the multidimensional elo ratings of the players based on the
    interactions against tasks rather than between each other.

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

    new_player_elos = deepcopy(player_elos)
    new_task_elos = deepcopy(task_elos)

    # Initialize U and V matrices
    u_matrix = np.array([e.vector for e in player_elos])
    v_matrix = np.array([e.vector for e in task_elos])

    omega = _build_omega(k)

    player_indices = {p: i for i, p in enumerate(players)}
    task_indices = {t: i for i, t in enumerate(tasks)}

    for interac in interactions:
        p0_id = player_indices[interac.players[0]]
        p1_id = task_indices[interac.players[1]]
        rating0 = new_player_elos[p0_id].mu
        rating1 = new_task_elos[p1_id].mu

        adjustment_matrix = u_matrix @ omega @ v_matrix.T
        p1_adjustment = adjustment_matrix[p0_id, p1_id]

        # Expected win probability
        win_prob = sigmoid(rating0 - rating1 + p1_adjustment)

        # Delta between expected and actual win
        # I had to change the index here, and I don't know why
        # I am so confused
        delta = interac.outcomes[1] - win_prob

        # Update ratings. r has higher lr than c
        new_player_elos[p0_id].mu += lr1*delta
        new_task_elos[p1_id].mu -= lr1*delta

        tmp_u_mat = np.array(u_matrix)
        tmp_v_mat = np.array(v_matrix)

        tmp_u_mat[p0_id] = \
            u_matrix[p0_id] + lr2 * delta * (omega @ v_matrix[p1_id]).T
        tmp_u_mat[p1_id] = \
            u_matrix[p1_id] - lr2 * delta * (omega @ u_matrix[p0_id]).T

        u_matrix = tmp_u_mat
        v_matrix = tmp_v_mat

        new_player_elos[p0_id].vector = list(u_matrix[p0_id])
        new_task_elos[p1_id].vector = list(v_matrix[p1_id])

    return new_player_elos, new_task_elos
