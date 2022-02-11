from typing import Callable, Hashable, Sequence, Optional, TypeVar

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def similarity(
    s1: S1, s2: S2, *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...
