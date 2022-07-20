from typing import (
    Any,
    Collection,
    Mapping,
    Tuple,
    Callable,
    Hashable,
    Sequence,
    Iterable,
    Optional,
    Union,
    overload,
    TypeVar,
    List,
    Generator,
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

try:
    import numpy as np

    def cdist(
        queries: Iterable[_S1],
        choices: Iterable[_S2],
        *,
        scorer: Callable[..., _ResultType] = ratio,
        processor: Optional[Callable[..., _StringType]] = None,
        score_cutoff: Optional[_ResultType] = None,
        dtype: Optional[np.dtype] = None,
        workers: int = 1,
        **kwargs: Any
    ) -> np.ndarray: ...

except ImportError:
    pass
