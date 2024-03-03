
import numpy as np
import scipy
import nashpy
from more_itertools import collapse

from popcore import Interaction

from poprank.utils import to_pairwise
from ...core import Rate


_ERROR_UNSUPPORTED_NASH_METHOD = """
    Unsupported Nash method {}
"""
_ERROR_UNSUPPORTED_NASH_SELECTION_METHOD = """
    Unsupported equilibrium selection method {}
"""
_ERROR_NASH_NOT_FOUND = """
    Nash equilibrium not found with {}, try a different method.
"""


class EmpiricalPayoffMatrix:

    def __init__(
        self,
        players: "list[str]",
        interactions: "list[Interaction]"
    ) -> None:
        self._dim = len(players)
        self._players = players

        self._idxs = {player: idx for idx, player in enumerate(players)}
        self._epm = np.zeros(shape=(self._dim, self._dim))

        self._populate_epm(interactions)

    def __array__(self):
        return self._epm

    def rectify(self):
        self._epm = np.maximum(self._epm, 0)

    def _populate_epm(
        self, interactions: "list[Interaction]"
    ):
        for interaction in interactions:
            player_1, player_2 = interaction.players
            outcome_1, outcome_2 = interaction.outcomes
            if outcome_1 > outcome_2:
                self._epm[self._idxs[player_1], self._idxs[player_2]] += 1.0
            elif outcome_2 > outcome_1:
                self._epm[self._idxs[player_2], self._idxs[player_1]] += 1.0

        self._epm += 1.0
        self._epm /= self._epm + self._epm.T
        self._epm = np.log(self._epm)


def _compute_szs_meta_nash(
    empirical_game: nashpy.Game, nash_method: str = "linear"
):
    """Computes the Nash equilibrium of the empirical game
    using one of the solvers in `nashpy`.

    :param empirical_game: _description_
    :type empirical_game: nashpy.Game
    :param nash_method: _description_, defaults to "linear"
    :type nash_method: str, optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    nashs = None
    match nash_method:
        case "vertex":
            nashs = empirical_game.vertex_enumeration()
        case "linear":
            nashs = empirical_game.linear_program()
        case "lemke_howson":
            nashs = empirical_game.lemke_howson(0)
        case "lemke_howson_enum":
            nashs = empirical_game.lemke_howson_enumeration()
        case _:
            raise ValueError(
                _ERROR_UNSUPPORTED_NASH_METHOD.format(nash_method)
            )
    nashs = [nash for nash in nashs]
    if len(nashs) == 0:
        raise ValueError(
            _ERROR_NASH_NOT_FOUND.format(nash_method)
        )

    # TODO: Inconsistent library behavior between methods
    if nash_method == "linear" or nash_method == "lemke_howson":
        nashs = [(nashs)]

    return nashs


def nash_avg(
    players: "list[str]", interactions: "list[Interaction]",
    rates: list[Rate] = None, nash_method: "str" = "linear",
) -> "list[Rate]":
    """Computes the Nash Average of the players based on the interactions.

    This method of rating is non-transitive.
    Based on https://arxiv.org/abs/1806.02643.

    :param list[str] players: A list containing all unique player identifiers.
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly 2
        players and be zero-sum.
    :param str nash_method: The method used to compute the nash equilibriums.
        Can be one of 'vertex', 'linear', 'lemke_howson' or
        'lemke_howson_enum'. Defaults to 'vertex'.
    :param str nash_selection: The method used to select the nash equilibrium
        among the possible options. Defaults to 'max_entropy'.

    :returns: The nash average for an antisymmetric zero-sum payoff matrix
        built from the interactions.
    :rtype: list[Rate]

    Example
    -------

    .. code-block:: python

        # Example using rock-paper-scissor

        from poprank.functional import nash_avg
        from poprank import Rate
        from popcore import Interaction

        nash = nash_avg(
            players=["r", "p", "s"],
            interactions=[
                Interaction(
                    players=["r", "p"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["p", "s"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["s", "r"],
                    outcomes=[-1.0, 1.0]
                ),
                Interaction(
                    players=["r", "r"],
                    outcomes=[0.0, 0.0],
                ),
                Interaction(
                    players=["p", "p"],
                    outcomes=[0.0, 0.0]
                ),
                Interaction(
                    players=["s", "s"],
                    outcomes=[0.0, 0.0]
                )
            ]
        )

        # nash should be equal to
        # [
        #    Rate(1/3),
        #    Rate(1/3),
        #    Rate(1/3)
        # ]


    .. seealso::
        :class:`poprank.rates.Rate`

        :meth:`poprank.functional.nash_avgAvT`

        :meth:`poprank.functional.rectified_nash_avg`
    """

    if rates is not None:
        print("nashavg (warning): initial rates not supported")

    empirical_payoff_matrix = EmpiricalPayoffMatrix(
        players, to_pairwise(interactions)
    )
    empirical_game = nashpy.Game(
        empirical_payoff_matrix
    )

    nashs = _compute_szs_meta_nash(empirical_game, nash_method)

    player_1_nash, _ = nashs[0]

    return [Rate(value) for value in player_1_nash]


def rectified_nash_avg(
    players: "list[str]", interactions: "list[Interaction]", rates: list[Rate] = None,
    nash_method: "str" = "linear", nash_selection: "str" = "max_entropy"
) -> "list[Rate]":
    """Computes the rectified Nash Average of the players based on the
    interactions.

    This method of rating is non-transitive.
    Based on https://arxiv.org/pdf/1901.08106.pdf.

    :param list[str] players: A list containing all unique player identifiers.
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly two
        players and be zero-sum.
    :param str nash_method: The method used to compute the nash equilibriums.
        Can be one of 'vertex', 'linear', 'lemke_howson' or
        'lemke_howson_enum'. Defaults to 'vertex'.
    :param str nash_selection: The method used to select the nash equilibrium
        among the possible options. Defaults to 'max_entropy'.

    :returns: The nash average for a zero-sum payoff matrix built
        from the interactions.
    :rtype: list[Rate]

    Example
    -------

    TODO

    .. seealso::
        :class:`poprank.rates.Rate`

        :meth:`poprank.functional.nash_avg`

        :meth:`poprank.functional.nash_avgAvT`
    """
    try:
        import nashpy
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrix(
        players, interactions
    )

    # Apply ReLU
    empirical_payoff_matrix.rectify()

    empirical_game = nashpy.Game(empirical_payoff_matrix)

    nashs = _compute_szs_meta_nash(empirical_game, nash_method)

    # verify that for each Nash, the Nash of each player
    # is the same (Nash of a Population against itself is unique)
    # see Re-evaluating Evaluation.
    player_1_nash, _ = nashs[0]

    return [Rate(value) for value in player_1_nash]

