# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Collection, Hashable, Sequence

from rapidfuzz.fuzz import ratio
from rapidfuzz.process_cpp_impl import FLOAT32 as _FLOAT32
from rapidfuzz.process_cpp_impl import FLOAT64 as _FLOAT64
from rapidfuzz.process_cpp_impl import INT8 as _INT8
from rapidfuzz.process_cpp_impl import INT16 as _INT16
from rapidfuzz.process_cpp_impl import INT32 as _INT32
from rapidfuzz.process_cpp_impl import INT64 as _INT64
from rapidfuzz.process_cpp_impl import UINT8 as _UINT8
from rapidfuzz.process_cpp_impl import UINT16 as _UINT16
from rapidfuzz.process_cpp_impl import UINT32 as _UINT32
from rapidfuzz.process_cpp_impl import UINT64 as _UINT64
from rapidfuzz.process_cpp_impl import cdist as _cdist
from rapidfuzz.process_cpp_impl import extract, extract_iter, extractOne

__all__ = ["extract", "extract_iter", "extractOne", "cdist"]

if TYPE_CHECKING:
    import numpy as np


def _dtype_to_type_num(dtype: np.dtype | None) -> int | None:
    import numpy as np

    if dtype is None:
        return None
    if dtype is np.int32:
        return _INT32
    if dtype is np.int8:
        return _INT8
    if dtype is np.int16:
        return _INT16
    if dtype is np.int64:
        return _INT64
    if dtype is np.uint8:
        return _UINT8
    if dtype is np.uint16:
        return _UINT16
    if dtype is np.uint32:
        return _UINT32
    if dtype is np.uint64:
        return _UINT64
    if dtype is np.float32:
        return _FLOAT32
    if dtype is np.float64:
        return _FLOAT64

    msg = "unsupported dtype"
    raise TypeError(msg)


def cdist(
    queries: Collection[Sequence[Hashable] | None],
    choices: Collection[Sequence[Hashable] | None],
    *,
    scorer: Callable[..., int | float] = ratio,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: int | float | None = None,
    score_hint: int | float | None = None,
    dtype: np.dtype | None = None,
    workers: int = 1,
    **kwargs: Any,
) -> np.ndarray:
    import numpy as np

    dtype = _dtype_to_type_num(dtype)
    return np.asarray(
        _cdist(
            queries,
            choices,
            scorer=scorer,
            processor=processor,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
            dtype=dtype,
            workers=workers,
            **kwargs,
        )
    )
