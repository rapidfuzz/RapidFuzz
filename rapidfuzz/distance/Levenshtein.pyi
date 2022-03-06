from typing import Callable, Hashable, Sequence, Optional, TypeVar, Tuple
from rapidfuzz.distance import Editops, Opcodes

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def distance(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None) -> int: ...

def normalized_distance(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def similarity(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None) -> int: ...

def normalized_similarity(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def editops(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None) -> Editops: ...

def opcodes(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None) -> Opcodes: ...
