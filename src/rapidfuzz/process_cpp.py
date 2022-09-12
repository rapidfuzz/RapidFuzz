# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from rapidfuzz.process_cpp_impl import (
    FLOAT32 as _FLOAT32,
    FLOAT64 as _FLOAT64,
    INT8 as _INT8,
    INT16 as _INT16,
    INT32 as _INT32,
    INT64 as _INT64,
    UINT8 as _UINT8,
    UINT16 as _UINT16,
    UINT32 as _UINT32,
    UINT64 as _UINT64,
    cdist as _cdist,
    extract as extract,
    extractOne as extractOne,
    extract_iter as extract_iter,
)

from rapidfuzz.fuzz import ratio as _ratio


def _dtype_to_type_num(dtype):
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

    raise TypeError("unsupported dtype")


def cdist(
    queries,
    choices,
    *,
    scorer=_ratio,
    processor=None,
    score_cutoff=None,
    dtype=None,
    workers=1,
    **kwargs
):
    import numpy as np

    dtype = _dtype_to_type_num(dtype)
    return np.asarray(
        _cdist(
            queries,
            choices,
            scorer=scorer,
            processor=processor,
            score_cutoff=score_cutoff,
            dtype=dtype,
            workers=workers,
            **kwargs
        )
    )
