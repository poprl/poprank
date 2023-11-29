from popcore import Interaction
from poprank import MeloRate
import numpy as np
from copy import deepcopy


def _build_omega(k):
    omega = np.zeros((2*k, 2*k))
    e = np.atleast_3d(np.identity(2*k))
    for i in range(k):
        omega += e[:, 2*i] @ e[:, 2*i+1].T - e[2*i+1] @ e[2*i].T
    return omega


def _sigmoid(x):
    return 1/(1+np.exp(-x))


def mElo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[MeloRate]", k: int = 1, lr1: float = 16, lr2: float = 1
) -> "list[MeloRate]":

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

    for i, interac in enumerate(interactions):
        p0_id = player_indices[interac.players[0]]
        p1_id = player_indices[interac.players[1]]
        rating0 = new_elos[p0_id].mu
        rating1 = new_elos[p1_id].mu

        adjustment_matrix = c_matrix @ omega @ c_matrix.T
        p1_adjustment = adjustment_matrix[p0_id, p1_id]

        # Expected win probability
        win_prob = _sigmoid(rating0 - rating1 + p1_adjustment)

        # Delta between expected and actual win
        delta = interac.outcomes[0] - win_prob

        # Update ratings. r has higher lr than c
        new_elos[p0_id].mu += lr1*delta
        new_elos[p1_id].mu -= lr1*delta

        tmp_c_mat = np.array(c_matrix)

        tmp_c_mat[p0_id] = \
            c_matrix[p0_id] + lr2 * delta * (omega @ c_matrix[p1_id]).T
        tmp_c_mat[p1_id] = \
            c_matrix[p1_id] - lr2 * delta * (omega @ c_matrix[p0_id]).T

        c_matrix = tmp_c_mat

        new_elos[p0_id].vector = list(c_matrix[p0_id])
        new_elos[p1_id].vector = list(c_matrix[p1_id])

    return new_elos
