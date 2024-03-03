
from popcore import Interaction
from poprank import Rate
from poprank.functional.rates.nashavg import _compute_szs_meta_nash


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


def nashavg(
    players: "list[str]", tasks: "list[str]",
    interactions: "list[Interaction]",
    nash_method: "str" = "linear",
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
    except ImportError as e:
        raise e  # TODO: improve this exception handling

    empirical_payoff_matrix = EmpiricalPayoffMatrixAvT(
        players, tasks, interactions
    )
    empirical_game = nashpy.Game(
        empirical_payoff_matrix.__array__(),
        -empirical_payoff_matrix.__array__())

    nashs = _compute_szs_meta_nash(empirical_game, nash_method)

    nash = nashs[0]

    player_nash = [Rate(value) for value in nash[0]]
    task_nash = [Rate(value) for value in nash[1]]
    return player_nash, task_nash
