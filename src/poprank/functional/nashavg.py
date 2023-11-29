from popcore import Interaction
from poprank import Rate

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


def nash_avg(
    players: "list[str]", interactions: "list[Interaction]",
    nash_method: "str" = "vertex", nash_selection: "str" = "max_entropy"
) -> "list[Rate]":
    """

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

    # verify that for each Nash, the Nash of each player
    # is the same (Nash of a Population against itself is unique)
    # see Re-evaluating Evaluation.

    population_nash = []
    for nash in nashs:
        player_1_nash, player_2_nash = nash
        if not np.allclose(player_1_nash, player_2_nash):
            raise ValueError()
        population_nash.append(player_1_nash)

    if len(population_nash) == 1:
        nash = population_nash[0]
    else:
        nash_idx = None
        match nash_selection:
            # TODO: Should entropy selection be part of the Nash finding optim?
            # or is this the right way
            case "max_entropy":
                nash_idx = np.argmax(
                    np.array([
                        scipy.stats.entropy(nash) for nash in population_nash
                    ])
                )
                nash = population_nash[nash_idx]
            case _:
                raise ValueError(
                    _ERROR_UNSUPPORTED_NASH_SELECTION_METHOD.format(
                        nash_selection
                        )
                )

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
    """

    """
    try:
        import nashpy
        import scipy
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrixAvT(
        players, tasks, interactions
    )
    empirical_game = nashpy.Game(empirical_payoff_matrix)

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

    # verify that for each Nash, the Nash of each player
    # is the same (Nash of a Population against itself is unique)
    # see Re-evaluating Evaluation.

    population_nash = []
    for nash in nashs:
        player, tasks = nash
        population_nash.append((player, tasks))

    if len(population_nash) == 1:
        nash = population_nash[0]
    else:
        nash_idx = None
        match nash_selection:
            # TODO: Should entropy selection be part of the Nash finding optim?
            # or is this the right way
            case "max_entropy":
                nash_idx = np.argmax(
                    np.array([
                        scipy.stats.entropy(np.array(nash).flatten)
                        for nash in population_nash
                    ])
                )
                nash = population_nash[nash_idx]
            case _:
                raise ValueError(
                    _ERROR_UNSUPPORTED_NASH_SELECTION_METHOD.format(
                        nash_selection
                        )
                )

    player_nash = [Rate(value) for value in nash[0]]
    task_nash = [Rate(value) for value in nash[1]]
    return player_nash, task_nash


def rectified_nash_avg(
    players: "list[str]", interactions: "list[Interaction]",
    ratings: "list[Rate]"
):
    """

    """
    try:
        import nashpy  # noqa
    except ImportError:
        raise

    # turn interactions into a win loose draw matrix
    # rectify the matrix (ReLU)
    # compute the nash, and if multiple, the highest entropy
    #
