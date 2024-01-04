import numpy as np
import networkx as nx
import scipy

from popcore import Interaction, Population
from ...core import Rate


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

    graph = nx.from_numpy_array(payoff, create_using=nx.DiGraph)

    laplacian = nx.directed_laplacian_matrix(graph)

    ranking = scipy.linalg.null_space(laplacian)
    assert ranking.shape[-1] == 1

    ranking = np.squeeze(ranking)
    return ranking * np.sign(np.max(ranking))  # svd sign correction 
