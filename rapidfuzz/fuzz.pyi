from typing import Callable, Hashable, Sequence, Optional, Union, overload, TypeVar
from rapidfuzz.utils import default_process
from rapidfuzz.distance import ScoreAlignment

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_ratio_alignment(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> ScoreAlignment: ...

def token_sort_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def token_set_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def token_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_sort_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_set_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def WRatio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...

def QRatio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[..., _StringType]] = default_process,
    score_cutoff: Optional[float] = 0) -> float: ...
