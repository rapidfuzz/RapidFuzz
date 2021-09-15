# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.utils import default_process

from rapidfuzz.string_metric import (
    levenshtein,
    normalized_levenshtein,
    hamming,
    normalized_hamming,
    jaro_similarity,
    jaro_winkler_similarity,
)

from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio,
    token_ratio,
    partial_token_sort_ratio,
    partial_token_set_ratio,
    partial_token_ratio,
    QRatio,
    WRatio
)

from libcpp.vector cimport vector
from libcpp.utility cimport move
from libc.stdint cimport uint8_t, int32_t
from libc.math cimport floor

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF

from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence, default_process_func

from array import array

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

cdef inline proc_string conv_sequence(seq) except *:
    if is_valid_string(seq):
        return move(convert_string(seq))
    elif isinstance(seq, array):
        return move(hash_array(seq))
    else:
        return move(hash_sequence(seq))

cdef extern from "cpp_process.hpp":
    cdef cppclass CachedScorerContext:
        CachedScorerContext()
        double ratio(const proc_string&, double) nogil except +

    cdef cppclass CachedDistanceContext:
        CachedDistanceContext()
        size_t ratio(const proc_string&, size_t) nogil except +

    # normalized distances
    # fuzz
    CachedScorerContext cached_ratio_init(                   const proc_string&, int) nogil except +
    CachedScorerContext cached_partial_ratio_init(           const proc_string&, int) except +
    CachedScorerContext cached_token_sort_ratio_init(        const proc_string&, int) except +
    CachedScorerContext cached_token_set_ratio_init(         const proc_string&, int) except +
    CachedScorerContext cached_token_ratio_init(             const proc_string&, int) except +
    CachedScorerContext cached_partial_token_sort_ratio_init(const proc_string&, int) except +
    CachedScorerContext cached_partial_token_set_ratio_init( const proc_string&, int) except +
    CachedScorerContext cached_partial_token_ratio_init(     const proc_string&, int) except +
    CachedScorerContext cached_WRatio_init(                  const proc_string&, int) except +
    CachedScorerContext cached_QRatio_init(                  const proc_string&, int) except +
    # string_metric
    CachedScorerContext cached_normalized_levenshtein_init(  const proc_string&, int, size_t, size_t, size_t) except +
    CachedScorerContext cached_normalized_hamming_init(      const proc_string&, int) except +
    CachedScorerContext cached_jaro_winkler_similarity_init( const proc_string&, int, double) except +
    CachedScorerContext cached_jaro_similarity_init(         const proc_string&, int) except +

    # distances
    CachedDistanceContext cached_levenshtein_init(const proc_string&, int, size_t, size_t, size_t) except +
    CachedDistanceContext cached_hamming_init(    const proc_string&, int) except +


cdef inline CachedScorerContext CachedNormalizedLevenshteinInit(const proc_string& query, int def_process, dict kwargs):
    cdef size_t insertion, deletion, substitution
    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    return move(cached_normalized_levenshtein_init(query, def_process, insertion, deletion, substitution))

cdef inline CachedDistanceContext CachedLevenshteinInit(const proc_string& query, int def_process, dict kwargs):
    cdef size_t insertion, deletion, substitution
    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    return move(cached_levenshtein_init(query, def_process, insertion, deletion, substitution))

cdef inline CachedScorerContext CachedJaroWinklerSimilarityInit(const proc_string& query, int def_process, dict kwargs):
    cdef double prefix_weight
    prefix_weight = kwargs.get("prefix_weight", 0.1)
    return move(cached_jaro_winkler_similarity_init(query, def_process, prefix_weight))

cdef inline int IsIntegratedScorer(object scorer):
    return (
        scorer is ratio or
        scorer is partial_ratio or
        scorer is token_sort_ratio or
        scorer is token_set_ratio or
        scorer is token_ratio or
        scorer is partial_token_sort_ratio or
        scorer is partial_token_set_ratio or
        scorer is partial_token_ratio or
        scorer is WRatio or
        scorer is QRatio or
        scorer is normalized_levenshtein or
        scorer is normalized_hamming or
        scorer is jaro_similarity or
        scorer is jaro_winkler_similarity
    )

cdef inline int IsIntegratedDistance(object scorer):
    return (
        scorer is levenshtein or
        scorer is hamming
    )

cdef inline CachedScorerContext CachedScorerInit(object scorer, const proc_string& query, int def_process, dict kwargs):
    cdef CachedScorerContext context

    if scorer is ratio:
        context = cached_ratio_init(query, def_process)
    elif scorer is partial_ratio:
        context = cached_partial_ratio_init(query, def_process)
    elif scorer is token_sort_ratio:
        context = cached_token_sort_ratio_init(query, def_process)
    elif scorer is token_set_ratio:
        context = cached_token_set_ratio_init(query, def_process)
    elif scorer is token_ratio:
        context = cached_token_ratio_init(query, def_process)
    elif scorer is partial_token_sort_ratio:
        context = cached_partial_token_sort_ratio_init(query, def_process)
    elif scorer is partial_token_set_ratio:
        context = cached_partial_token_set_ratio_init(query, def_process)
    elif scorer is partial_token_ratio:
        context = cached_partial_token_ratio_init(query, def_process)
    elif scorer is WRatio:
        context = cached_WRatio_init(query, def_process)
    elif scorer is QRatio:
        context = cached_QRatio_init(query, def_process)
    elif scorer is normalized_levenshtein:
        context = CachedNormalizedLevenshteinInit(query, def_process, kwargs)
    elif scorer is normalized_hamming:
        context = cached_normalized_hamming_init(query, def_process)
    elif scorer is jaro_similarity:
        context = cached_jaro_similarity_init(query, def_process)
    elif scorer is jaro_winkler_similarity:
        context = CachedJaroWinklerSimilarityInit(query, def_process, kwargs)

    return move(context)

cdef inline CachedDistanceContext CachedDistanceInit(object scorer, const proc_string& query, int def_process, dict kwargs):
    cdef CachedDistanceContext context

    if scorer is levenshtein:
        context = CachedLevenshteinInit(query, def_process, kwargs)
    elif scorer is hamming:
        context = cached_hamming_init(query, def_process)

    return move(context)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cdist_two_lists_similarity(
    const vector[proc_string]& queries,
    const vector[proc_string]& choices,
    scorer, score_cutoff, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t choices_len = choices.size()
    cdef size_t i, j
    cdef double c_score_cutoff = 0
    cdef np.ndarray[np.uint8_t, ndim=2] matrix = np.empty((queries_len, choices_len), dtype=np.uint8)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    c_score_cutoff = floor(c_score_cutoff)

    for i in range(queries_len):
        ScorerContext = CachedScorerInit(scorer, queries[i], 0, kwargs)
        for j in range(choices_len):
            matrix[i, j] = <uint8_t>floor(ScorerContext.ratio(choices[j], c_score_cutoff))

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cdist_two_lists_distance(
    const vector[proc_string]& queries, const vector[proc_string]& choices,
    scorer, score_cutoff, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t choices_len = choices.size()
    cdef size_t i, j
    cdef size_t c_max = <size_t>-1
    cdef np.ndarray[np.int32_t, ndim=2] matrix = np.empty((queries_len, choices_len), dtype=np.int32)

    if score_cutoff is not None and score_cutoff != -1:
        c_max = score_cutoff

    for i in range(queries_len):
        DistanceContext = CachedDistanceInit(scorer, queries[i], 0, kwargs)
        for j in range(choices_len):
            matrix[i, j] = <int32_t>DistanceContext.ratio(choices[j], c_max)

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline py_cdist_two_lists(
    const vector[PyObject*]& queries, const vector[PyObject*]& choices,
    scorer, score_cutoff, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t choices_len = choices.size()
    cdef size_t i, j
    cdef double c_score_cutoff = 0
    cdef np.ndarray[np.uint8_t, ndim=2] matrix = np.empty((queries_len, choices_len), dtype=np.uint8)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    c_score_cutoff = floor(c_score_cutoff)

    kwargs["processor"] = None
    kwargs["score_cutoff"] = c_score_cutoff

    for i in range(queries_len):
        for j in range(choices_len):
            matrix[i, j] = <uint8_t>floor(
                <double>scorer(<object>queries[i], <object>choices[j],**kwargs))

    return matrix

cdef cdist_two_lists(queries, choices, scorer, processor, score_cutoff, dict kwargs):
    cdef vector[proc_string] proc_queries
    cdef vector[proc_string] proc_choices
    cdef vector[PyObject*] proc_py_queries
    cdef vector[PyObject*] proc_py_choices
    cdef size_t queries_len = <size_t>len(queries)
    cdef size_t choices_len = <size_t>len(choices)

    try:
        if IsIntegratedScorer(scorer) or IsIntegratedDistance(scorer):
            proc_queries.reserve(queries_len)
            proc_choices.reserve(choices_len)

            # processor None/False
            if not processor:
                for query in queries:
                    proc_queries.push_back(move(conv_sequence(query)))

                for choice in choices:
                    proc_choices.push_back(move(conv_sequence(choice)))
            # processor has to be called through python
            elif processor is not default_process and callable(processor):
                proc_py_queries.reserve(queries_len)
                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)
                    proc_queries.push_back(move(conv_sequence(proc_query)))

                proc_py_choices.reserve(choices_len)
                for choice in choices:
                    proc_choice = processor(choice)
                    Py_INCREF(proc_choice)
                    proc_py_choices.push_back(<PyObject*>proc_choice)
                    proc_choices.push_back(move(conv_sequence(proc_choice)))

            # processor is True / default_process
            else:
                for query in queries:
                    proc_queries.push_back(
                        move(default_process_func(move(conv_sequence(query))))
                    )

                for choice in choices:
                    proc_choices.push_back(
                        move(default_process_func(move(conv_sequence(choice))))
                    )

            if IsIntegratedScorer(scorer):
                return cdist_two_lists_similarity(proc_queries, proc_choices, scorer, score_cutoff, kwargs)

            if IsIntegratedDistance(scorer):
                return cdist_two_lists_distance(proc_queries, proc_choices, scorer, score_cutoff, kwargs)

        else:
            proc_py_queries.reserve(queries_len)
            proc_py_choices.reserve(choices_len)

            # processor None/False
            if not processor:
                for query in queries:
                    Py_INCREF(query)
                    proc_py_queries.push_back(<PyObject*>query)

                for choice in choices:
                    Py_INCREF(choice)
                    proc_py_choices.push_back(<PyObject*>choice)
            # processor has to be called through python
            else:
                if not callable(processor):
                    processor = default_process

                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)

                for choice in choices:
                    proc_choice = processor(choice)
                    Py_INCREF(proc_choice)
                    proc_py_choices.push_back(<PyObject*>proc_choice)

            return py_cdist_two_lists(proc_py_queries, proc_py_choices, scorer, score_cutoff, kwargs)

    finally:
        # decref all reference counts
        for item in proc_py_queries:
            Py_DECREF(<object>item)

        for item in proc_py_choices:
            Py_DECREF(<object>item)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cdist_single_list_similarity(
    const vector[proc_string]& queries, scorer, score_cutoff, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t i, j
    cdef double c_score_cutoff = 0
    cdef np.ndarray[np.uint8_t, ndim=2] matrix = np.empty((queries_len, queries_len), dtype=np.uint8)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    c_score_cutoff = floor(c_score_cutoff)

    for i in range(queries_len):
        matrix[i, i] = 100
        ScorerContext = CachedScorerInit(scorer, queries[i], 0, kwargs)
        for j in range(i + 1, queries_len):
            score = <uint8_t>floor(ScorerContext.ratio(queries[j], c_score_cutoff))
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cdist_single_list_distance(
    const vector[proc_string]& queries, scorer, score_cutoff, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t i, j
    cdef size_t c_max = <size_t>-1
    cdef np.ndarray[np.int32_t, ndim=2] matrix = np.empty((queries_len, queries_len), dtype=np.int32)

    if score_cutoff is not None and score_cutoff != -1:
        c_max = score_cutoff

    for i in range(queries_len):
        matrix[i, i] = 0
        DistanceContext = CachedDistanceInit(scorer, queries[i], 0, kwargs)
        for j in range(i + 1, queries_len):
            score = <int32_t>DistanceContext.ratio(queries[j], c_max)
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix

cdef cdist_single_list(queries, scorer, processor, score_cutoff, dict kwargs):
    cdef size_t queries_len = <size_t>len(queries)

    cdef vector[proc_string] proc_queries
    cdef vector[PyObject*] proc_py_queries

    try:
        if IsIntegratedScorer(scorer) or IsIntegratedDistance(scorer):
            proc_queries.reserve(queries_len)

            # processor None/False
            if not processor:
                for query in queries:
                    proc_queries.push_back(move(conv_sequence(query)))
            # processor has to be called through python
            elif processor is not default_process and callable(processor):
                proc_py_queries.reserve(queries_len)
                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)
                    proc_queries.push_back(move(conv_sequence(proc_query)))

            # processor is True / default_process
            else:
                for query in queries:
                    proc_queries.push_back(
                        move(default_process_func(move(conv_sequence(query))))
                    )

            if IsIntegratedScorer(scorer):
                return cdist_single_list_similarity(proc_queries, scorer, score_cutoff, kwargs)

            if IsIntegratedDistance(scorer):
                return cdist_single_list_distance(proc_queries, scorer, score_cutoff, kwargs)

        else:
            proc_py_queries.reserve(queries_len)

            # processor None/False
            if not processor:
                for query in queries:
                    Py_INCREF(query)
                    proc_py_queries.push_back(<PyObject*>query)
            # processor has to be called through python
            else:
                if not callable(processor):
                    processor = default_process

                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)

            # scorer(a, b) might not be equal to scorer(b, a)
            return py_cdist_two_lists(proc_py_queries, proc_py_queries, scorer, score_cutoff, kwargs)

    finally:
        # decref all reference counts
        for item in proc_py_queries:
            Py_DECREF(<object>item)

def cdist(queries, choices, *, scorer=ratio, processor=None, score_cutoff=None, **kwargs):
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
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance).
        Custom functions are not supported so far!
        fuzz.ratio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance <= score_cutoff` are inserted as -1. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity >= score_cutoff` are inserted as 0.
        Default is None, which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    List[Tuple[Sequence[Hashable], Any, Any]]
    """
    if queries is choices:
        return cdist_single_list(queries, scorer, processor, score_cutoff, kwargs)
    else:
        return cdist_two_lists(queries, choices, scorer, processor, score_cutoff, kwargs)