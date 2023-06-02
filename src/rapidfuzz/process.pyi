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
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")
_ResultType = int | float

@overload
def extractOne(
    query: _S1,
    choices: Iterable[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> tuple[_S2, _ResultType, int]: ...
@overload
def extractOne(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> tuple[_S2, _ResultType, Any]: ...
@overload
def extract(
    query: _S1,
    choices: Collection[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    limit: int | None = 5,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> list[tuple[_S2, _ResultType, int]]: ...
@overload
def extract(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    limit: int | None = 5,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> list[tuple[_S2, _ResultType, Any]]: ...
@overload
def extract_iter(
    query: _S1,
    choices: Iterable[_S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_S2, _ResultType, int], None, None]: ...
@overload
def extract_iter(
    query: _S1,
    choices: Mapping[Any, _S2],
    *,
    scorer: Callable[..., _ResultType] = WRatio,
    processor: Callable[..., _StringType] | None = None,
    score_cutoff: _ResultType | None = None,
    score_hint: _ResultType | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> Generator[tuple[_S2, _ResultType, Any], None, None]: ...

try:
    import numpy as np

    def cdist(
        queries: Iterable[_S1],
        choices: Iterable[_S2],
        *,
        scorer: Callable[..., _ResultType] = ratio,
        processor: Callable[..., _StringType] | None = None,
        score_cutoff: _ResultType | None = None,
        score_hint: _ResultType | None = None,
        dtype: np.dtype | None = None,
        workers: int = 1,
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray: ...

except ImportError:
    pass
