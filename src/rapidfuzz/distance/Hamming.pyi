from typing import Callable, Hashable, Sequence, Optional, TypeVar
from rapidfuzz.distance import Editops, Opcodes

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

def distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
def normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
def normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def editops(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None
) -> Editops: ...
def opcodes(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None
) -> Opcodes: ...
