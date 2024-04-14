from __future__ import annotations

from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

from rapidfuzz.fuzz import WRatio, ratio

_StringType = Sequence[Hashable]
_StringType1 = TypeVar("_StringType1", bound=Sequence[Hashable])
_StringType2 = TypeVar("_StringType2", bound=Sequence[Hashable])
_KeyType = TypeVar("_KeyType")
_ResultType = TypeVar("_ResultType", int, float)

@overload
def extractOne(
    query: _StringType1 | None,
    choices: Mapping[_KeyType, _StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> tuple[_StringType2, _ResultType, _KeyType]: ...
@overload
def extractOne(
    query: _StringType1 | None,
    choices: Iterable[_StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> tuple[_StringType2, _ResultType, int]: ...
@overload
def extract(
    query: _StringType1 | None,
    choices: Mapping[_KeyType, _StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    limit: int | None = 5,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> list[tuple[_StringType2, _ResultType, _KeyType]]: ...
@overload
def extract(
    query: _StringType1 | None,
    choices: Collection[_StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    limit: int | None = 5,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> list[tuple[_StringType2, _ResultType, int]]: ...
@overload
def extract_iter(
    query: _StringType1 | None,
    choices: Iterable[_StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_StringType2, _ResultType, int], None, None]: ...
@overload
def extract_iter(
    query: _StringType1 | None,
    choices: Mapping[_KeyType, _StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_StringType2, _ResultType, _KeyType], None, None]: ...

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
    queries: Iterable[_StringType1],
    choices: Iterable[_StringType2],
    *,
    scorer: Callable[..., _ResultType] = ratio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    score_multiplier: _ResultType | None = None,
    dtype: int | None = None,
    workers: int = 1,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Any: ...
def cpdist(
    queries: Iterable[_StringType1],
    choices: Iterable[_StringType2],
    *,
    scorer: Callable[..., _ResultType] = ratio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    score_multiplier: _ResultType | None = None,
    dtype: int | None = None,
    workers: int = 1,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Any: ...
