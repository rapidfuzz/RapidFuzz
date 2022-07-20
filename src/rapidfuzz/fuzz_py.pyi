from typing import Callable, Hashable, Sequence, Optional, TypeVar, Any, Dict
from rapidfuzz.utils import default_process
from rapidfuzz.distance import ScoreAlignment
from typing_extensions import Protocol

class _ScorerAttributes(Protocol):
    _RF_ScorerPy: Dict

def _attr_decorator(func: Any) -> _ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

@_attr_decorator
def ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def partial_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def partial_ratio_alignment(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> Optional[ScoreAlignment]: ...
@_attr_decorator
def token_sort_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def token_set_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def token_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def partial_token_sort_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def partial_token_set_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def partial_token_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def WRatio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@_attr_decorator
def QRatio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
