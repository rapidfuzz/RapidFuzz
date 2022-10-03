# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

from typing import Hashable, Sequence

from rapidfuzz._utils import fallback_import as _fallback_import


def default_process(sentence: Sequence[Hashable]) -> Sequence[Hashable]:
    ...


default_process = _fallback_import("rapidfuzz.utils", "default_process")
