from typing import Callable, Hashable, Sequence, Optional, Union, TypeVar

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def distance(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None) -> int: ...

def normalized_distance(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def similarity(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None) -> int: ...

def normalized_similarity(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...
