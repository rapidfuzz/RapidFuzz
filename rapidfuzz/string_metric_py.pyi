from typing import (
    Callable,
    Hashable,
    Sequence,
    Optional,
    TypeVar,
    Tuple,
    List,
    Any,
    Dict,
)
from typing_extensions import Protocol

class ScorerAttributes(Protocol):
    _RF_ScorerPy: Dict

def attr_decorator(func: Any) -> ScorerAttributes:
    return func

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

@attr_decorator
def levenshtein(
    s1: S1,
    s2: S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    max: Optional[int] = None
) -> int: ...
@attr_decorator
def normalized_levenshtein(
    s1: S1,
    s2: S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def levenshtein_editops(
    s1: S1, s2: S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> List[Tuple[str, int, int]]: ...
@attr_decorator
def hamming(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    max: Optional[int] = None
) -> int: ...
@attr_decorator
def normalized_hamming(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def jaro_similarity(
    s1: S1,
    s2: S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
@attr_decorator
def jaro_winkler_similarity(
    s1: S1,
    s2: S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0
) -> float: ...
