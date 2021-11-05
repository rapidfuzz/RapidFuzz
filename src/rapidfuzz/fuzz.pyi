from typing import Callable, Hashable, Sequence, Optional, Union, overload, TypeVar

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    score_cutoff: Optional[float] = 0) -> float: ...

def token_sort_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def token_set_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def token_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_sort_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_set_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def partial_token_ratio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def WRatio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...

def QRatio(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]],
    score_cutoff: Optional[float] = 0) -> float: ...
