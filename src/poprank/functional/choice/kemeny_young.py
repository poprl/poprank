from typing import List
from ..._core import Rank
from ...excepts import raise_with_message_code


def kemeny_young(votes: List[Rank]) -> Rank:
    raise_with_message_code(
        "not_implemented", NotImplementedError, "kemeny_young"
    )
