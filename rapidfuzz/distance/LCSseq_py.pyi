from typing import Callable, Hashable, Sequence, Optional, TypeVar, Any, Dict
from rapidfuzz.distance import Editops, Opcodes
from typing_extensions import Protocol

class ScorerAttributes(Protocol):
    _RF_ScorerPy7: Dict

def attr_decorator(func: Any) -> ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

@attr_decorator
def distance(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@attr_decorator
def normalized_distance(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def similarity(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@attr_decorator
def normalized_similarity(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def editops(
    s1: S1, s2: S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Editops: ...
def opcodes(
    s1: S1, s2: S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Opcodes: ...
