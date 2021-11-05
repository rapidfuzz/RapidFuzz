# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.fuzz import ratio
from rapidfuzz.utils import default_process

from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp cimport bool
from libc.math cimport floor

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF
from cython.operator cimport dereference

from cpp_common cimport (
    RF_StringWrapper, RF_KwargsWrapper, KwargsInit,
    is_valid_string, convert_string, hash_array, hash_sequence
)

from array import array
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
cimport cython

from rapidfuzz_capi cimport (
    RF_Kwargs, RF_String, RF_Scorer, RF_DistanceInit, RF_SimilarityInit,
    RF_SIMILARITY, RF_DISTANCE
)
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

np.import_array()

cdef inline RF_String conv_sequence(seq) except *:
    if is_valid_string(seq):
        return move(convert_string(seq))
    elif isinstance(seq, array):
        return move(hash_array(seq))
    else:
        return move(hash_sequence(seq))

cdef extern from "cpp_process_cdist.hpp":
    object cdist_single_list_distance_impl(  const RF_KwargsWrapper&, RF_DistanceInit, const vector[RF_StringWrapper]&, int, int, size_t) except +
    object cdist_single_list_similarity_impl(const RF_KwargsWrapper&, RF_SimilarityInit, const vector[RF_StringWrapper]&, int, int, double) except +
    object cdist_two_lists_distance_impl(    const RF_KwargsWrapper&, RF_DistanceInit, const vector[RF_StringWrapper]&, const vector[RF_StringWrapper]&, int, int, size_t) except +
    object cdist_two_lists_similarity_impl(  const RF_KwargsWrapper&, RF_SimilarityInit, const vector[RF_StringWrapper]&, const vector[RF_StringWrapper]&, int, int, double) except +
    void set_score_similarity(np.PyArrayObject*, int, np.npy_intp, np.npy_intp, double)

cdef extern from "cpp_utils.hpp":
    RF_String default_process_func(RF_String sentence) except +

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
    const vector[RF_StringWrapper]& queries,
    const vector[RF_StringWrapper]& choices,
    RF_Scorer scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef double c_score_cutoff = 0
    cdef RF_KwargsWrapper kwargs_context = KwargsInit(scorer, kwargs)
    cdef int c_dtype = dtype_to_type_num_similarity(dtype)
    cdef int c_workers = workers

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    return cdist_two_lists_similarity_impl(kwargs_context, scorer.scorer.similarity_init, queries, choices, c_dtype, c_workers, c_score_cutoff)

cdef inline cdist_two_lists_distance(
    const vector[RF_StringWrapper]& queries, const vector[RF_StringWrapper]& choices,
    RF_Scorer scorer, max_, dtype, workers, dict kwargs
):
    cdef size_t c_max = <size_t>-1
    cdef RF_KwargsWrapper kwargs_context = KwargsInit(scorer, kwargs)
    cdef int c_dtype = dtype_to_type_num_distance(dtype)
    cdef int c_workers = workers

    if max_ is not None:
        if max_ < -1:
            raise TypeError("max has to be >= -1")
        c_max = max_

    return cdist_two_lists_distance_impl(kwargs_context, scorer.scorer.distance_init, queries, choices, c_dtype, c_workers, c_max)

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
    cdef vector[RF_StringWrapper] proc_queries
    cdef vector[RF_StringWrapper] proc_choices
    cdef vector[PyObject*] proc_py_queries
    cdef vector[PyObject*] proc_py_choices
    cdef size_t queries_len = <size_t>len(queries)
    cdef size_t choices_len = <size_t>len(choices)
    cdef RF_Scorer* scorer_context

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    try:
        if scorer_context:
            proc_queries.reserve(queries_len)
            proc_choices.reserve(choices_len)

            # processor None/False
            if not processor:
                for query in queries:
                    proc_queries.push_back(move(RF_StringWrapper(conv_sequence(query))))

                for choice in choices:
                    proc_choices.push_back(move(RF_StringWrapper(conv_sequence(choice))))
            # processor has to be called through python
            elif processor is not default_process and callable(processor):
                proc_py_queries.reserve(queries_len)
                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)
                    proc_queries.push_back(move(RF_StringWrapper(conv_sequence(proc_query))))

                proc_py_choices.reserve(choices_len)
                for choice in choices:
                    proc_choice = processor(choice)
                    Py_INCREF(proc_choice)
                    proc_py_choices.push_back(<PyObject*>proc_choice)
                    proc_choices.push_back(move(RF_StringWrapper(conv_sequence(proc_choice))))

            # processor is True / default_process
            else:
                for query in queries:
                    proc_queries.push_back(
                        move(RF_StringWrapper(default_process_func(conv_sequence(query))))
                    )

                for choice in choices:
                    proc_choices.push_back(
                        move(RF_StringWrapper(default_process_func(conv_sequence(choice))))
                    )

            if scorer_context.scorer_type == RF_SIMILARITY:
                return cdist_two_lists_similarity(proc_queries, proc_choices, dereference(scorer_context), score_cutoff, dtype, workers, kwargs)
            else:
                return cdist_two_lists_distance(proc_queries, proc_choices, dereference(scorer_context), score_cutoff, dtype, workers, kwargs)

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
    const vector[RF_StringWrapper]& queries, RF_Scorer scorer, score_cutoff, dtype, workers, dict kwargs
):
    cdef double c_score_cutoff = 0
    cdef RF_KwargsWrapper kwargs_context = KwargsInit(scorer, kwargs)
    cdef int c_dtype = dtype_to_type_num_similarity(dtype)
    cdef int c_workers = workers

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
    if c_score_cutoff < 0 or c_score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    return cdist_single_list_similarity_impl(kwargs_context, scorer.scorer.similarity_init, queries, c_dtype, c_workers, c_score_cutoff)

cdef inline cdist_single_list_distance(
    const vector[RF_StringWrapper]& queries, RF_Scorer scorer, max_, dtype, workers, dict kwargs
):
    cdef size_t c_max = <size_t>-1
    cdef RF_KwargsWrapper kwargs_context = KwargsInit(scorer, kwargs)
    cdef int c_dtype = dtype_to_type_num_distance(dtype)
    cdef int c_workers = workers

    if max_ is not None:
        if max_ < -1:
            raise TypeError("max has to be >= -1")
        c_max = max_

    return cdist_single_list_distance_impl(kwargs_context, scorer.scorer.distance_init, queries, c_dtype, c_workers, c_max)

cdef cdist_single_list(queries, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
    cdef size_t queries_len = <size_t>len(queries)

    cdef vector[RF_StringWrapper] proc_queries
    cdef vector[PyObject*] proc_py_queries
    cdef RF_Scorer* scorer_context

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    try:
        if scorer_context:
            proc_queries.reserve(queries_len)

            # processor None/False
            if not processor:
                for query in queries:
                    proc_queries.push_back(move(RF_StringWrapper(conv_sequence(query))))
            # processor has to be called through python
            elif processor is not default_process and callable(processor):
                proc_py_queries.reserve(queries_len)
                for query in queries:
                    proc_query = processor(query)
                    Py_INCREF(proc_query)
                    proc_py_queries.push_back(<PyObject*>proc_query)
                    proc_queries.push_back(move(RF_StringWrapper(conv_sequence(proc_query))))

            # processor is True / default_process
            else:
                for query in queries:
                    proc_queries.push_back(
                        move(RF_StringWrapper(default_process_func(conv_sequence(query))))
                    )

            if scorer_context.scorer_type == RF_SIMILARITY:
                return cdist_single_list_similarity(proc_queries, dereference(scorer_context), score_cutoff, dtype, workers, kwargs)
            else:
                return cdist_single_list_distance(proc_queries, dereference(scorer_context), score_cutoff, dtype, workers, kwargs)
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
