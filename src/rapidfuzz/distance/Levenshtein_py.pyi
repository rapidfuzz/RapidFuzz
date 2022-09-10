from typing import Callable, Hashable, Sequence, Optional, TypeVar, Tuple, Any, Dict
from rapidfuzz.distance import Editops, Opcodes
from typing_extensions import Protocol

class _ScorerAttributes(Protocol):
    _RF_ScorerPy: Dict

def _attr_decorator(func: Any) -> _ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

@_attr_decorator
def distance(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@_attr_decorator
def normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def similarity(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
@_attr_decorator
def normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def editops(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_hint: Optional[int] = None
) -> Editops: ...
def opcodes(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_hint: Optional[int] = None
) -> Opcodes: ...
