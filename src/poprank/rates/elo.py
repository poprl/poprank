from typing import Any, List
from popcore import Interaction, Population
from ..core import RateModule
from ..functional.rates.elo import elo, EloRate


class Elo(RateModule[EloRate]):
    """_summary_

    :param RateModule: _description_
    :type RateModule: _type_
    """
    def __init__(
        self, population: Population, rates: List[EloRate],
        default_rate: float, k_factor: float = 20.0, wdl: bool = False,
        reduction: str = "aggregate"
    ):
        """
            #TODO: Complete

        :param k_factor: _description_, defaults to 20
        :type k_factor: float, optional
        :param wdl: _description_, defaults to False
        :type wdl: bool, optional
        :param reduction: _description_, defaults to "aggregate"
        :type reduction: str, optional
        """
        super().__init__(population, rates, default_rate)
        self._k_factor = k_factor
        self._wdl = wdl
        self._reduction = reduction

    def _rate(
        self, interactions: List[Interaction], **kwds: Any
    ) -> List[EloRate]:
        return elo(
            self._population.players, interactions, elos=self.rates,
            k_factor=kwds.get('k_factor', self._k_factor),
            wdl=kwds.get('wdl', self._wdl),
            reduce=kwds.get('reduce', self._reduction)
        )
