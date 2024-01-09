import numpy as np
import networkx as nx
import scipy

from popcore import Interaction, Population, History
from ..._core import Rate


# Devlin, Stephen, and Thomas Treloar.
# “A Network Diffusion Ranking Family That Includes the Methods of Markov, Massey, and Colley.”
# Journal of Quantitative Analysis in Sports, vol. 14, no. 3, Sept. 2018
# pp. 91–101, https://doi.org/10.1515/jqas-2017-0098.

def laplacian(
    population: Population, interactions: list[Interaction],
    ratings: list[Rate]
) -> list[Rate]:
    """
        TODO: Implement
    """
    history = History.from_interactions(interactions)
    payoff = history.to_payoff_matrix()

    graph = nx.from_numpy_array(payoff, create_using=nx.DiGraph)

    laplacian = nx.directed_laplacian_matrix(graph)

    rates = scipy.linalg.null_space(laplacian)
    assert rates.shape[-1] == 1

    rates = np.squeeze(rates)
    rates *= np.sign(np.max(rates))  # svd sign correction

    return [Rate(rate) for rate in rates]
