# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

from rapidfuzz._utils import fallback_import as _fallback_import
from rapidfuzz.distance import ScoreAlignment
from rapidfuzz.utils import default_process


def ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = 0,
) -> float:
    ...


def partial_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = 0,
) -> float:
    ...


def partial_ratio_alignment(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = 0,
) -> ScoreAlignment | None:
    ...


def token_sort_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def token_set_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def token_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def partial_token_sort_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def partial_token_set_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def partial_token_ratio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def WRatio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def QRatio(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = default_process,
    score_cutoff: float | None = 0,
) -> float:
    ...


def _GetScorerFlagsSimilarity(**kwargs: Any) -> dict[str, Any]:
    return {"optimal_score": 100, "worst_score": 0, "flags": (1 << 5)}


_fuzz_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _GetScorerFlagsSimilarity
}

_mod = "rapidfuzz.fuzz"
ratio = _fallback_import(_mod, "ratio", cached_scorer_call=_fuzz_attribute)
partial_ratio = _fallback_import(
    _mod, "partial_ratio", cached_scorer_call=_fuzz_attribute
)
partial_ratio_alignment = _fallback_import(
    _mod, "partial_ratio_alignment", cached_scorer_call=_fuzz_attribute
)
token_sort_ratio = _fallback_import(
    _mod, "token_sort_ratio", cached_scorer_call=_fuzz_attribute
)
token_set_ratio = _fallback_import(
    _mod, "token_set_ratio", cached_scorer_call=_fuzz_attribute
)
token_ratio = _fallback_import(_mod, "token_ratio", cached_scorer_call=_fuzz_attribute)
partial_token_sort_ratio = _fallback_import(
    _mod, "partial_token_sort_ratio", cached_scorer_call=_fuzz_attribute
)
partial_token_set_ratio = _fallback_import(
    _mod, "partial_token_set_ratio", cached_scorer_call=_fuzz_attribute
)
partial_token_ratio = _fallback_import(
    _mod, "partial_token_ratio", cached_scorer_call=_fuzz_attribute
)
WRatio = _fallback_import(_mod, "WRatio", cached_scorer_call=_fuzz_attribute)
QRatio = _fallback_import(_mod, "QRatio", cached_scorer_call=_fuzz_attribute)
