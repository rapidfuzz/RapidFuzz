from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from rapidfuzz.fuzz import WRatio, ratio

_StringType = Sequence[Hashable]
_AnyStringType = TypeVar("_AnyStringType", bound=_StringType)
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")
_ResultType = Union[int, float]

@overload
def extractOne(
    query: _S1,
    choices: Iterable[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> Tuple[_S2, _ResultType, int]: ...
@overload
def extractOne(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> Tuple[_S2, _ResultType, Any]: ...
@overload
def extract(
    query: _S1,
    choices: Collection[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    limit: Optional[int] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> List[Tuple[_S2, _ResultType, int]]: ...
@overload
def extract(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> List[Tuple[_S2, _ResultType, Any]]: ...
@overload
def extract_iter(
    query: _S1,
    choices: Iterable[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[_S2, _ResultType, int], None, None]: ...
@overload
def extract_iter(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[_S2, _ResultType, Any], None, None]: ...

FLOAT32: int
FLOAT64: int
INT8: int
INT16: int
INT32: int
INT64: int
UINT8: int
UINT16: int
UINT32: int
UINT64: int

def cdist(
    queries: Iterable[_S1],
    choices: Iterable[_S2],
    *,
    scorer: Callable[..., _ResultType] = ratio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[_ResultType] = None,
    dtype: Optional[int] = None,
    workers: int = 1,
    **kwargs: Any
) -> Any: ...
