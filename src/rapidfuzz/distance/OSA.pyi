# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

from typing import Callable, Hashable, Sequence

def distance(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def normalized_distance(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = 0,
) -> float: ...
def similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def normalized_similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = 0,
) -> float: ...
