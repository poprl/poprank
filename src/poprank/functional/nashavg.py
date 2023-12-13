from popcore import Interaction
from poprank import Rate
from more_itertools import collapse

import numpy as np


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
            if len(interaction.players) != 2:
                raise ValueError("")
            if sum(interaction.outcomes) != 0:
                raise ValueError("Nash average requires a zero sum game")
            player_1, player_2 = interaction.players
            self._epm[self._idxs[player_1], self._idxs[player_2]] += \
                interaction.outcomes[0]
            self._epm[self._idxs[player_2], self._idxs[player_1]] += \
                interaction.outcomes[1]


class EmpiricalPayoffTensor:
    pass


def _compute_nashs(empirical_game, nash_method):
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

    # Inconsistent library behavior between methods
    if nash_method == "linear" or nash_method == "lemke_howson":
        nashs = [(nashs)]

    return nashs


def _select_max_entropy(population_nash, nash_selection):
    import scipy

    if len(population_nash) == 1:
        return population_nash[0]

    nash_idx = None
    match nash_selection:
        # TODO: Should entropy selection be part of the Nash finding optim?
        # or is this the right way
        case "max_entropy":
            nash_idx = np.argmax(
                np.array([
                    [
                        scipy.stats.entropy(list(collapse(nash)))
                        for nash in population_nash
                    ]])
            )
            nash = population_nash[nash_idx]
        case _:
            raise ValueError(
                _ERROR_UNSUPPORTED_NASH_SELECTION_METHOD.format(
                    nash_selection
                    )
            )

    return nash


def nash_avg(
    players: "list[str]", interactions: "list[Interaction]",
    nash_method: "str" = "vertex", nash_selection: "str" = "max_entropy"
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
    try:
        import nashpy
        import scipy
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrix(
        players, interactions
    )
    empirical_game = nashpy.Game(empirical_payoff_matrix)

    nashs = _compute_nashs(empirical_game, nash_method)

    # verify that for each Nash, the Nash of each player
    # is the same (Nash of a Population against itself is unique)
    # see Re-evaluating Evaluation.

    population_nash = []
    for nash in nashs:
        player_1_nash, player_2_nash = nash
        if not np.allclose(player_1_nash, player_2_nash):
            raise ValueError()
        population_nash.append(player_1_nash)

    nash = _select_max_entropy(population_nash, nash_selection)

    return [Rate(value) for value in nash]


class EmpiricalPayoffMatrixAvT:

    def __init__(
        self,
        players: "list[str]",
        tasks: "list[str]",
        interactions: "list[Interaction]"
    ) -> None:
        self._dim = (len(players), len(tasks))
        self._players = players
        self.tasks = tasks

        self._pidxs = {player: idx for idx, player in enumerate(players)}
        self._tidxs = {task: idx for idx, task in enumerate(tasks)}
        self._epm = np.zeros(shape=self._dim)

        self._populate_epm(interactions)

    def __array__(self):
        return self._epm

    def _populate_epm(
        self, interactions: "list[Interaction]"
    ):
        for interaction in interactions:
            if len(interaction.players) != 2:
                raise ValueError("")
            player, task = interaction.players
            self._epm[self._pidxs[player], self._tidxs[task]] += \
                interaction.outcomes[0]


def nash_avgAvT(
    players: "list[str]", tasks: "list[str]",
    interactions: "list[Interaction]",
    nash_method: "str" = "vertex",
    nash_selection: "str" = "max_entropy"
) -> "tuple[list[Rate]]":
    """Computes the Nash Average of the players against the tasks based on the
    interactions.

    This method of rating is non-transitive.
    Based on https://arxiv.org/abs/1806.02643.

    :param list[str] players: A list containing all unique player identifiers.
    :param list[str] tasks: A list containing all unique task identifiers.
    :param list[Interaction] interactions: A list containing the interactions
        to get a rating from. Every interaction should be between exactly one
        player and one task, in this order, and be zero-sum.
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

    .. code-block:: python

        from poprank.functional import nash_avgAvT
        from poprank import Rate
        from popcore import Interaction

        players = ["a", "b", "c"]
        tasks = ["d", "e"]
        interac = [
            Interaction(["a", "d"], [1, 0]),
            Interaction(["b", "d"], [0, 1]),
            Interaction(["c", "d"], [1, 0]),
            Interaction(["a", "e"], [0, 1]),
            Interaction(["b", "e"], [1, 0]),
            Interaction(["c", "e"], [0, 1])
        ]

        player_nash, task_nash = nash_avgAvT(
            players, tasks, interac, nash_method="lemke_howson_enum")


    .. seealso::
        :class:`poprank.rates.Rate`

        :meth:`poprank.functional.nash_avg`

        :meth:`poprank.functional.rectified_nash_avg`
    """
    try:
        import nashpy
        import scipy
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrixAvT(
        players, tasks, interactions
    )
    empirical_game = nashpy.Game(
        empirical_payoff_matrix.__array__(),
        -empirical_payoff_matrix.__array__())

    nashs = _compute_nashs(empirical_game, nash_method)

    population_nash = []
    for nash in nashs:
        player, tasks = nash
        population_nash.append((player, tasks))

    nash = _select_max_entropy(population_nash, nash_selection)

    player_nash = [Rate(value) for value in nash[0]]
    task_nash = [Rate(value) for value in nash[1]]
    return player_nash, task_nash


def rectified_nash_avg(
    players: "list[str]", interactions: "list[Interaction]",
    nash_method: "str" = "vertex", nash_selection: "str" = "max_entropy"
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
        import scipy
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrix(
        players, interactions
    )

    # Apply ReLU
    empirical_payoff_matrix.rectify()

    empirical_game = nashpy.Game(empirical_payoff_matrix)

    nashs = _compute_nashs(empirical_game, nash_method)

    # verify that for each Nash, the Nash of each player
    # is the same (Nash of a Population against itself is unique)
    # see Re-evaluating Evaluation.

    population_nash = []
    for nash in nashs:
        player_1_nash, player_2_nash = nash
        if not np.allclose(player_1_nash, player_2_nash):
            raise ValueError()
        population_nash.append(player_1_nash)

    nash = _select_max_entropy(population_nash, nash_selection)

    return [Rate(value) for value in nash]

    # turn interactions into a win loose draw matrix
    # rectify the matrix (ReLU)
    # compute the nash, and if multiple, the highest entropy
    #
