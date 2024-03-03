from typing import List, Optional
from functools import partial
import numpy as np
import networkx as nx
import scipy

from popcore import Interaction, Player, Population
from poprank import Rate
from poprank.utils import to_win_matrix


# Devlin, Stephen, and Thomas Treloar.
# “A Network Diffusion Ranking Family That Includes the Methods of Markov,
# Massey, and Colley.”
# Journal of Quantitative Analysis in Sports, vol. 14, no. 3, Sept. 2018
# pp. 91–101, https://doi.org/10.1515/jqas-2017-0098.

def laplacian(
    players: list[Player],
    interactions: list[Interaction],
    rates: Optional[List[Rate]] = None,
) -> List[Rate]:
    """_summary_

    :param interactions: _description_
    :type interactions: list[Interaction]
    :param population: _description_, defaults to None
    :type population: Optional[Population], optional
    :param rates: _description_, defaults to None
    :type rates: Optional[List[Rate]], optional
    :param reduction: which matrix is used to constrct the pairwise
        interaction graph. One of "payoff | av_payoff | win | win_rates
        | preference"defaults to "payoff"
    :type reduction: Optional[str], optional
    :return: _description_
    :rtype: list[Rate]
    """

    if rates is not None:
        print("laplacian (warning): initial rates not supported")

    graph = nx.from_numpy_array(
        to_win_matrix(
            interactions,
            Population.from_players_uid(None, players),
            normalize=True
        ),
        create_using=nx.DiGraph
    )

    laplacian = nx.directed_laplacian_matrix(graph)

    rates = scipy.linalg.null_space(laplacian)
    if rates.shape[-1] != 1:
        ValueError("laplacian.indetermined")

    rates = np.squeeze(rates)
    # rates *= np.sign(np.max(rates))  # svd sign correction

    return [Rate(rate) for rate in rates]
