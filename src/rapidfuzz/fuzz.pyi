from typing import Callable, Hashable, Optional, Sequence, TypeVar

from rapidfuzz.distance import ScoreAlignment
from rapidfuzz.utils import default_process

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

def ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def partial_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def partial_ratio_alignment(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> Optional[ScoreAlignment]: ...
def token_sort_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def token_set_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def token_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def partial_token_sort_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def partial_token_set_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def partial_token_ratio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def WRatio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
def QRatio(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0
) -> float: ...
