from typing import Callable, Hashable, Sequence, Optional, TypeVar, Any, Dict
from typing_extensions import Protocol

class _ScorerAttributes(Protocol):
    _RF_ScorerPy: Dict

def _attr_decorator(func: Any) -> _ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

@_attr_decorator
def distance(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@_attr_decorator
def normalized_distance(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def similarity(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@_attr_decorator
def normalized_similarity(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
