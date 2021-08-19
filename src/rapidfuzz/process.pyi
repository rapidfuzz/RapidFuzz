from typing import Any, Mapping, Tuple, Callable, Hashable, Sequence, Iterable, Optional, Union, overload, TypeVar, List, Generator
from rapidfuzz.fuzz import WRatio

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")
ResultType = TypeVar("ResultType", int, float)

@overload
def extractOne(
    query: _StringType,
    choices: Iterable[_StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[_StringType, ResultType, int]: ...

@overload
def extractOne(
    query: _StringType,
    choices: Mapping[Any, _StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[_StringType, ResultType, Any]: ...

@overload
def extractOne(
    query: S1,
    choices: Iterable[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[S2, ResultType, int]: ...

@overload
def extractOne(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Tuple[S2, ResultType, Any]: ...

@overload
def extract(
    query: _StringType,
    choices: Iterable[_StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    limit: Optional[int] = ...,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[_StringType, ResultType, int]]: ...


@overload
def extract(
    query: _StringType,
    choices: Mapping[Any, _StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    limit: Optional[int] = ...,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[_StringType, ResultType, Any]]: ...

@overload
def extract(
    query: S1,
    choices: Iterable[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    limit: Optional[int] = ...,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[S2, ResultType, int]]: ...

@overload
def extract(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> List[Tuple[S2, ResultType, Any]]: ...


@overload
def extract_iter(
    query: _StringType,
    choices: Iterable[_StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[_StringType, ResultType, int], None, None]: ...

@overload
def extract_iter(
    query: _StringType,
    choices: Mapping[Any, _StringType], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Optional[bool] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[_StringType, ResultType, Any], None, None]: ...

@overload
def extract_iter(
    query: S1,
    choices: Iterable[S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[S2, ResultType, int], None, None]: ...

@overload
def extract_iter(
    query: S1,
    choices: Mapping[Any, S2], *,
    scorer: Callable[..., ResultType] = WRatio,
    processor: Callable[[Union[S1, S2]], _StringType] = None,
    score_cutoff: Optional[ResultType] = None,
    **kwargs: Any
) -> Generator[Tuple[S2, ResultType, Any], None, None]: ...

