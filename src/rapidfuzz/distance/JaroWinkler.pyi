from typing import Callable, Hashable, Sequence, Optional, Union, TypeVar

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def similarity(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...
