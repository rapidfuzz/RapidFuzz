from typing import Callable, Hashable, Sequence, Optional, TypeVar, Any, Dict
from rapidfuzz.utils import default_process
from rapidfuzz.distance import ScoreAlignment
from typing_extensions import Protocol

class ScorerAttributes(Protocol):
    _RF_ScorerPy: Dict

def attr_decorator(func: Any) -> ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

@attr_decorator
def ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def partial_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def partial_ratio_alignment(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> ScoreAlignment: ...
@attr_decorator
def token_sort_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def token_set_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def token_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def partial_token_sort_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def partial_token_set_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def partial_token_ratio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def WRatio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def QRatio(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
