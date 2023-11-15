from popcore import Interaction
from poprank import MeloRate
import numpy as np
from copy import deepcopy


def _build_omega(k):
    omega = np.zeros((2*k, 2*k))
    for i in range(k):
        e_2i_1 = np.zeros((2*k, 1))
        e_2i_1[2*i, 0] = 1
        e_2i = np.zeros((2*k, 1))
        e_2i[2*i+1, 0] = 1
        omega += e_2i_1 @ e_2i.T - e_2i @ e_2i_1.T
    return omega


def _sigmoid(x):
    return 1/(1+np.exp(-x))


def mElo(
    players: "list[str]", interactions: "list[Interaction]",
    elos: "list[MeloRate]", k: int = 1, lr1: float = 16, lr2: float = 1,
    iterations: int = 10
) -> "list[MeloRate]":

    new_elos = deepcopy(elos)

    assert len(players) == len(new_elos), "Elos and players must match"
    for e in new_elos:
        assert e.k == k, "K value missmatch"

    # Outcomes must be in interval [0, 1]
    # k_factor must be positive int

    # Perhaps decompose an observed WIN/LOSS matrix in to a C Omega C'
    # for better initial params?
    # Inititalize C matrix
    # c_matrix = np.array([e.vector for e in elos]).T
    # c_matrix = np.zeros((2*k, len(players)))
    c_matrix = np.ones((2*k, len(players)))

    omega = _build_omega(k)

    player_indices = {p: i for i, p in enumerate(players)}

    for iter in range(iterations):
        for i, interac in enumerate(interactions):
            p0_id = player_indices[interac.players[0]]
            p1_id = player_indices[interac.players[1]]
            rating0 = new_elos[p0_id].mu
            rating1 = new_elos[p1_id].mu

            adjustment_matrix = c_matrix.T @ omega @ c_matrix
            p1_adjustment = adjustment_matrix[p0_id, p1_id]

            # Expected win proba
            win_prob = _sigmoid(rating0 - rating1 + p1_adjustment)

            # Delta between expected and actual win
            delta = interac.outcomes[0] - win_prob

            # Update ratings. r has higher lr than c
            new_elos[p0_id].mu += lr1*delta
            new_elos[p1_id].mu -= lr1*delta

            c_matrix[:, p0_id] += lr2 * delta * (omega @ c_matrix[:, p0_id]).T
            c_matrix[:, p1_id] -= lr2 * delta * (omega @ c_matrix[:, p1_id]).T

            new_elos[p0_id].vector = list(c_matrix[:, p0_id])
            new_elos[p1_id].vector = list(c_matrix[:, p1_id])

    return new_elos
