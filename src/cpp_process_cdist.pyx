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
from cython.operator cimport dereference

from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence, default_process_func

from array import array
from libc.stdlib cimport malloc, free

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

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cdef struct LevenshteinWeightTable:
        size_t insert_cost
        size_t delete_cost
        size_t replace_cost

cdef extern from "cpp_process.hpp":
    cdef cppclass CachedScorerContext:
        CachedScorerContext()
        double ratio(const proc_string&, double) nogil except +

    cdef cppclass CachedDistanceContext:
        CachedDistanceContext()
        size_t ratio(const proc_string&, size_t) nogil except +

    ctypedef void (*context_deinit) (void* context)

    cdef cppclass KwargsContext:
        KwargsContext()

        void* context
        context_deinit deinit

    ctypedef KwargsContext (*kwargs_context_init)(dict kwargs) except *
    ctypedef CachedDistanceContext (*distance_context_init)(const KwargsContext& kwargs, const proc_string& str) nogil except +
    ctypedef CachedScorerContext (*scorer_context_init)(const KwargsContext& kwargs, const proc_string& str) nogil except +

    cdef struct DistanceFunctionTable:
        kwargs_context_init kwargs_init
        distance_context_init init

    cdef struct ScorerFunctionTable:
        kwargs_context_init kwargs_init
        scorer_context_init init

    # normalized distances
    # fuzz
    ScorerFunctionTable CreateRatioFunctionTable() nogil except +
    ScorerFunctionTable CreatePartialRatioFunctionTable() nogil except +
    ScorerFunctionTable CreateTokenSortRatioFunctionTable() nogil except +
    ScorerFunctionTable CreateTokenSetRatioFunctionTable() nogil except +
    ScorerFunctionTable CreateTokenRatioFunctionTable() nogil except +
    ScorerFunctionTable CreatePartialTokenSortRatioFunctionTable() nogil except +
    ScorerFunctionTable CreatePartialTokenSetRatioFunctionTable() nogil except +
    ScorerFunctionTable CreatePartialTokenRatioFunctionTable() nogil except +
    ScorerFunctionTable CreateWRatioFunctionTable() nogil except +
    ScorerFunctionTable CreateQRatioFunctionTable() nogil except +

    # string_metric

    CachedScorerContext cached_jaro_winkler_similarity_init(const KwargsContext& kwargs, const proc_string& str) nogil except +
    CachedScorerContext cached_normalized_levenshtein_init(const KwargsContext& kwargs, const proc_string& str) nogil except +
    ScorerFunctionTable CreateNormalizedHammingFunctionTable()
    ScorerFunctionTable CreateJaroSimilarityFunctionTable()

    # distances
    DistanceFunctionTable CreateHammingFunctionTable()
    CachedDistanceContext cached_levenshtein_init(const KwargsContext& kwargs, const proc_string& str) nogil except +

cdef extern from "cpp_process_cdist.hpp":
    object cdist_single_list_distance_impl(const KwargsContext&, distance_context_init, const vector[proc_string]&, int, int, size_t) except +
    object cdist_single_list_similarity_impl(const KwargsContext&, scorer_context_init, const vector[proc_string]&, int, int, double) except +
    object cdist_two_lists_distance_impl(const KwargsContext&, distance_context_init, const vector[proc_string]&, const vector[proc_string]&, int, int, size_t) except +
    object cdist_two_lists_similarity_impl(const KwargsContext&, scorer_context_init, const vector[proc_string]&, const vector[proc_string]&, int, int, double) except +
    void set_score_similarity(np.PyArrayObject*, int, np.npy_intp, np.npy_intp, double)

cdef KwargsContext LevenshteinKwargsInit(dict kwargs) except *:
    cdef KwargsContext context
    cdef LevenshteinWeightTable* weights
    cdef size_t insertion, deletion, substitution
    weights = <LevenshteinWeightTable*>malloc(sizeof(LevenshteinWeightTable))

    if (NULL == weights):
        raise MemoryError

    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    dereference(weights).insert_cost = insertion
    dereference(weights).delete_cost = deletion
    dereference(weights).replace_cost = substitution
    context.context = weights
    context.deinit = free

    return move(context)

cdef inline ScorerFunctionTable CreateNormalizedLevenshteinFunctionTable():
    cdef ScorerFunctionTable table
    table.kwargs_init = &LevenshteinKwargsInit
    table.init = cached_normalized_levenshtein_init
    return table

cdef inline DistanceFunctionTable CreateLevenshteinFunctionTable():
    cdef DistanceFunctionTable table
    table.kwargs_init = &LevenshteinKwargsInit
    table.init = cached_levenshtein_init
    return table

cdef KwargsContext JaroWinklerKwargsInit(dict kwargs) except *:
    cdef KwargsContext context
    cdef double* prefix_weight
    prefix_weight = <double*>malloc(sizeof(double))

    if (NULL == prefix_weight):
        raise MemoryError

    prefix_weight[0] = kwargs.get("prefix_weight", 0.1)
    context.context = prefix_weight
    context.deinit = free

    return move(context)

cdef inline ScorerFunctionTable CreateJaroWinklerSimilarityFunctionTable():
    cdef ScorerFunctionTable table
    table.kwargs_init = &JaroWinklerKwargsInit
    table.init = cached_jaro_winkler_similarity_init
    return table

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

cdef inline ScorerFunctionTable CachedScorerInit(object scorer):
    cdef ScorerFunctionTable context

    if scorer is ratio:
        context = CreateRatioFunctionTable()
    elif scorer is partial_ratio:
        context = CreatePartialRatioFunctionTable()
    elif scorer is token_sort_ratio:
        context = CreateTokenSortRatioFunctionTable()
    elif scorer is token_set_ratio:
        context = CreateTokenSetRatioFunctionTable()
    elif scorer is token_ratio:
        context = CreateTokenRatioFunctionTable()
    elif scorer is partial_token_sort_ratio:
        context = CreatePartialTokenSortRatioFunctionTable()
    elif scorer is partial_token_set_ratio:
        context = CreatePartialTokenSetRatioFunctionTable()
    elif scorer is partial_token_ratio:
        context = CreatePartialTokenRatioFunctionTable()
    elif scorer is WRatio:
        context = CreateWRatioFunctionTable()
    elif scorer is QRatio:
        context = CreateQRatioFunctionTable()
    elif scorer is normalized_levenshtein:
        context = CreateNormalizedLevenshteinFunctionTable()
    elif scorer is normalized_hamming:
        context = CreateNormalizedHammingFunctionTable()
    elif scorer is jaro_similarity:
        context = CreateJaroSimilarityFunctionTable()
    elif scorer is jaro_winkler_similarity:
        context = CreateJaroWinklerSimilarityFunctionTable()

    return move(context)

cdef inline DistanceFunctionTable CachedDistanceInit(object scorer):
    cdef DistanceFunctionTable table

    if scorer is levenshtein:
        table = CreateLevenshteinFunctionTable()
    elif scorer is hamming:
        table = CreateHammingFunctionTable()

    return table

cdef int dtype_to_type_num_similarity(dtype) except -1:
    if dtype is None or dtype is np.uint8:
        return np.NPY_UINT8
    if dtype is np.float32:
        return np.NPY_FLOAT32
    if dtype is np.float64:
        return np.NPY_FLOAT64

    raise TypeError("invalid dtype (use np.uint8, np.float32 or np.float64)")

cdef int dtype_to_type_num_distance(dtype) except -1:
    if dtype is None or dtype is np.int32:
        return np.NPY_INT32
    if dtype is np.int8:
        return np.NPY_INT8
    if dtype is np.int16:
        return np.NPY_INT16
    if dtype is np.int64:
        return np.NPY_INT64

    raise TypeError("invalid dtype (use np.int8, np.int16, np.int32 or np.int64)")

cdef inline cdist_two_lists_similarity(
    const vector[proc_string]& queries,
    const vector[proc_string]& choices,
    scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef double c_score_cutoff = 0
    cdef ScorerFunctionTable table = CachedScorerInit(scorer)
    cdef KwargsContext kwargs_context
    cdef int c_dtype = dtype_to_type_num_similarity(dtype)
    cdef int c_workers = workers

    if (NULL != table.kwargs_init):
        kwargs_context = table.kwargs_init(kwargs)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    return cdist_two_lists_similarity_impl(kwargs_context, table.init, queries, choices, c_dtype, c_workers, c_score_cutoff)

cdef inline cdist_two_lists_distance(
    const vector[proc_string]& queries, const vector[proc_string]& choices,
    scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef size_t c_max = <size_t>-1
    cdef DistanceFunctionTable table = CachedDistanceInit(scorer)
    cdef KwargsContext kwargs_context
    cdef int c_dtype = dtype_to_type_num_distance(dtype)
    cdef int c_workers = workers

    if (NULL != table.kwargs_init):
        kwargs_context = table.kwargs_init(kwargs)

    if score_cutoff is not None and score_cutoff != -1:
        c_max = score_cutoff

    return cdist_two_lists_distance_impl(kwargs_context, table.init, queries, choices, c_dtype, c_workers, c_max)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline py_cdist_two_lists(
    const vector[PyObject*]& queries, const vector[PyObject*]& choices,
    scorer, score_cutoff, dtype, dict kwargs
):
    cdef size_t queries_len = queries.size()
    cdef size_t choices_len = choices.size()
    cdef size_t i, j
    cdef double c_score_cutoff = 0
    cdef int c_dtype = dtype_to_type_num_similarity(dtype)
    cdef double score
    matrix = np.empty((queries_len, choices_len), dtype=dtype)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    c_score_cutoff = floor(c_score_cutoff)

    kwargs["processor"] = None
    kwargs["score_cutoff"] = c_score_cutoff

    for i in range(queries_len):
        for j in range(choices_len):
            score = <double>scorer(<object>queries[i], <object>choices[j],**kwargs)
            set_score_similarity(<np.PyArrayObject*>matrix, c_dtype, i, j, score)

    return matrix

cdef cdist_two_lists(queries, choices, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
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
                return cdist_two_lists_similarity(proc_queries, proc_choices, scorer, score_cutoff, dtype, workers, kwargs)
       

            if IsIntegratedDistance(scorer):
                return cdist_two_lists_distance(proc_queries, proc_choices, scorer, score_cutoff, dtype, workers, kwargs)

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

            return py_cdist_two_lists(proc_py_queries, proc_py_choices, scorer, score_cutoff, dtype, kwargs)

    finally:
        # decref all reference counts
        for item in proc_py_queries:
            Py_DECREF(<object>item)

        for item in proc_py_choices:
            Py_DECREF(<object>item)

cdef inline cdist_single_list_similarity(
    const vector[proc_string]& queries, scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef double c_score_cutoff = 0
    cdef ScorerFunctionTable table = CachedScorerInit(scorer)
    cdef KwargsContext kwargs_context
    cdef int c_dtype = dtype_to_type_num_similarity(dtype)
    cdef int c_workers = workers

    if (NULL != table.kwargs_init):
        kwargs_context = table.kwargs_init(kwargs)

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    return cdist_single_list_similarity_impl(kwargs_context, table.init, queries, c_dtype, c_workers, c_score_cutoff)

cdef inline cdist_single_list_distance(
    const vector[proc_string]& queries, scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef size_t c_max = <size_t>-1
    cdef DistanceFunctionTable table = CachedDistanceInit(scorer)
    cdef KwargsContext kwargs_context
    cdef int c_dtype = dtype_to_type_num_distance(dtype)
    cdef int c_workers = workers

    if (NULL != table.kwargs_init):
        kwargs_context = table.kwargs_init(kwargs)

    if score_cutoff is not None and score_cutoff != -1:
        c_max = score_cutoff

    return cdist_single_list_distance_impl(kwargs_context, table.init, queries, c_dtype, c_workers, c_max)


cdef cdist_single_list(queries, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
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
                return cdist_single_list_similarity(proc_queries, scorer, score_cutoff, dtype, workers, kwargs)
       
            if IsIntegratedDistance(scorer):
                return cdist_single_list_distance(proc_queries, scorer, score_cutoff, dtype, workers, kwargs)

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
            return py_cdist_two_lists(proc_py_queries, proc_py_queries, scorer, score_cutoff, dtype, kwargs)

    finally:
        # decref all reference counts
        for item in proc_py_queries:
            Py_DECREF(<object>item)

def cdist(queries, choices, *, scorer=ratio, processor=None, score_cutoff=None, dtype=None, workers=1, **kwargs):
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
    dtype : data-type, optional
        The desired data-type for the result array. Depending on the scorer type the following
        dtypes are supported:
        - similarity: np.uint8, np.float32, np.float64
        - distance: np.int8, np.int16, np.int32, np.int64
        If not given, then the type will be np.uint8 for similarities and np.int32 for distances.
    workers : int, optional
        The calculation is subdivided into workers sections and evaluated in parallel.
        Supply -1 to use all available CPU cores.
        This argument is only available for scorers which are part of rapidfuzz. For custom
        scorers this has not effect.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    ndarray
        Returns a matrix of dtype with the distance/similarity between each pair
        of the two collections of inputs.
    """
    if queries is choices:
        return cdist_single_list(queries, scorer, processor, score_cutoff, dtype, workers, kwargs)
    else:
        return cdist_two_lists(queries, choices, scorer, processor, score_cutoff, dtype, workers, kwargs)
