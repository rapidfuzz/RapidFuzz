from typing import Callable, Hashable, Sequence, Optional, TypeVar, Tuple, List

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

def levenshtein(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    max: Optional[int] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
def normalized_levenshtein(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def levenshtein_editops(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> List[Tuple[str, int, int]]: ...
def hamming(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    max: Optional[int] = None,
    score_cutoff: Optional[int] = None
) -> int: ...
def normalized_hamming(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def jaro_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
def jaro_winkler_similarity(
    s1: _S1,
    s2: _S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
