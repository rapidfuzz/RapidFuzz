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
_ResultType = TypeVar("_ResultType")

@overload
def extractOne(
    query: _StringType1 | None,
    choices: Mapping[_KeyType, _StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: float | None = None,
    score_hint: float | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> tuple[_StringType2, _ResultType, _KeyType]: ...
@overload
def extractOne(
    query: _StringType1 | None,
    choices: Iterable[_StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: float | None = None,
    score_hint: float | None = None,
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
    score_cutoff: float | None = None,
    score_hint: float | None = None,
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
    score_cutoff: float | None = None,
    score_hint: float | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> list[tuple[_StringType2, _ResultType, int]]: ...
@overload
def extract_iter(
    query: _StringType1 | None,
    choices: Mapping[_KeyType, _StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: float | None = None,
    score_hint: float | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_StringType2, _ResultType, _KeyType], None, None]: ...
@overload
def extract_iter(
    query: _StringType1 | None,
    choices: Iterable[_StringType2 | None],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: float | None = None,
    score_hint: float | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_StringType2, _ResultType, int], None, None]: ...

try:
    import numpy as np

    def cdist(
        queries: Iterable[_StringType1],
        choices: Iterable[_StringType2],
        *,
        scorer: Callable[..., _ResultType] = ratio,
        processor: Callable[..., _StringType] | None = None,
        score_cutoff: float | None = None,
        score_hint: float | None = None,
        dtype: np.dtype | None = None,
        workers: int = 1,
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray: ...

    def cpdist(
        queries: Iterable[_StringType1],
        choices: Iterable[_StringType2],
        *,
        scorer: Callable[..., _ResultType] = ratio,
        processor: Callable[..., _StringType] | None = None,
        score_cutoff: float | None = None,
        score_hint: float | None = None,
        dtype: np.dtype | None = None,
        workers: int = 1,
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray: ...

except ImportError:
    pass
