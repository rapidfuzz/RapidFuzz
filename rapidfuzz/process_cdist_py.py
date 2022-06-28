# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
import numpy as np
from rapidfuzz.fuzz_py import ratio


def _dtype_to_type_num(dtype, scorer, **kwargs):
    if dtype is not None:
        return dtype

    params = getattr(scorer, "_RF_ScorerPy", None)
    if params is not None:
        flags = params["get_scorer_flags"](**kwargs)
        if flags["flags"] & (1 << 6):
            return np.int32
        else:
            return np.float32

    return np.float32


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
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance <= score_cutoff` are inserted as -1. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity >= score_cutoff` are inserted as 0.
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
    dtype = _dtype_to_type_num(dtype, scorer, **kwargs)
    results = np.zeros((len(queries), len(choices)), dtype=dtype)

    if queries is choices:
        if processor is None:
            proc_queries = queries
        else:
            proc_queries = [processor(x) for x in queries]

        for i, query in enumerate(proc_queries):
            results[i, i] = scorer(
                query, query, processor=None, score_cutoff=score_cutoff, **kwargs
            )
            for j in range(i + 1, len(proc_queries)):
                results[i, j] = results[j, i] = scorer(
                    query,
                    proc_queries[j],
                    processor=None,
                    score_cutoff=score_cutoff,
                    **kwargs
                )
    else:
        if processor is None:
            proc_queries = queries
            proc_choices = choices
        else:
            proc_queries = [processor(x) for x in queries]
            proc_choices = [processor(x) for x in choices]

        for i, query in enumerate(proc_queries):
            for j, choice in enumerate(proc_choices):
                results[i, j] = scorer(
                    query, choice, processor=None, score_cutoff=score_cutoff, **kwargs
                )

    return results
