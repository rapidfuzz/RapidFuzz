# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from rapidfuzz.process_cdist_cpp_impl import (
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
)

from rapidfuzz.fuzz import ratio

# numpy is required to use cdist
try:
    import numpy as np
except ImportError:
    pass


def _dtype_to_type_num(dtype):
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
    scorer=ratio,
    processor=None,
    score_cutoff=None,
    dtype=None,
    workers=1,
    **kwargs
):
    """
    Compute distance/similarity between each pair of the two collections of inputs.

    Parameters
    ----------
    queries : Collection[Sequence[Hashable]]
        list of all strings the queries
    choices : Collection[Sequence[Hashable]]
        list of all strings the query should be compared
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be:

        - a scorer using the RapidFuzz C-API like the builtin scorers in RapidFuzz,
          which can return a distance or similarity between two strings. Further details can be found here.
        - a Python function which returns a similarity between two strings in the range 0-100. This is not
          recommended, since it is far slower than a scorer using the RapidFuzz C-API.

        fuzz.ratio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold to be passed to the scorer.
        Default is None, which deactivates this behaviour.
    dtype : data-type, optional
        The desired data-type for the result array.Depending on the scorer type the following
        dtypes are supported:

        - similarity:
          - np.float32, np.float64
          - np.uint8 -> stores fixed point representation of the result scaled to a range 0-100
        - distance:
          - np.int8, np.int16, np.int32, np.int64

        If not given, then the type will be np.float32 for similarities and np.int32 for distances.
    workers : int, optional
        The calculation is subdivided into workers sections and evaluated in parallel.
        Supply -1 to use all available CPU cores.
        This argument is only available for scorers using the RapidFuzz C-API so far, since it
        releases the Python GIL.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    ndarray
        Returns a matrix of dtype with the distance/similarity between each pair
        of the two collections of inputs.
    """
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
