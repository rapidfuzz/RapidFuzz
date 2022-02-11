from typing import Any, Collection, Mapping, Tuple, Callable, Hashable, Sequence, Iterable, Optional, Union, overload, TypeVar, List, Generator
from rapidfuzz.fuzz import WRatio, ratio

_StringType = Sequence[Hashable]
_AnyStringType = TypeVar("_AnyStringType", bound=_StringType)
S1 = TypeVar("S1")
S2 = TypeVar("S2")
ResultType = Union[int, float]

@overload
def extractOne(
    query: S1,
    choices: Iterable[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[S2, ResultType, int]: ...

@overload
def extractOne(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[S2, ResultType, Any]: ...

@overload
def extract(
    query: S1,
    choices: Collection[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    limit: Optional[int] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[S2, ResultType, int]]: ...

@overload
def extract(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[S2, ResultType, Any]]: ...


@overload
def extract_iter(
    query: S1,
    choices: Iterable[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[S2, ResultType, int], None, None]: ...

@overload
def extract_iter(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[S2, ResultType, Any], None, None]: ...

try:
    import numpy as np

    def cdist(
        queries: Iterable[S1],
        choices: Iterable[S2], *,
        scorer: Callable[..., ResultType] = ratio,
        processor: Optional[Callable[..., _StringType]] = None,
        score_cutoff: Optional[ResultType] = None,
        dtype: Optional[np.dtype] = None,
        workers: int = 1,
        **kwargs: Any
    ) -> np.ndarray: ...

except ImportError:
    pass
