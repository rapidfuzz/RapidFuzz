from typing import Callable, Hashable, Sequence, Optional, Union, TypeVar, Tuple

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def levenshtein(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    max: Optional[int] = None) -> int: ...

def normalized_levenshtein(
    s1: S1, s2: S2, *,
    weights: Optional[Tuple[int, int, int]] = (1,1,1),
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    score_cutoff: Optional[float] = 0) -> float: ...

def levenshtein_editops(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType]) -> List[Tuple[str, int, int]]: ...

def hamming(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    max: Optional[int] = None) -> int: ...

def normalized_hamming(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    score_cutoff: Optional[float] = 0) -> float: ...

def jaro_similarity(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    score_cutoff: Optional[float] = 0) -> float: ...

def jaro_winkler_similarity(
    s1: S1, s2: S2, *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[[Union[S1, S2]]], _StringType],
    score_cutoff: Optional[float] = 0) -> float: ...
