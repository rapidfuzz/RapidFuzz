# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.fuzz import ratio
from rapidfuzz.utils import default_process

from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp cimport bool
from libc.math cimport floor
from libc.stdint cimport uint32_t, uint64_t, int64_t

cimport cython
from cython.operator cimport dereference

from cpp_common cimport (
    PyObjectWrapper, RF_StringWrapper, RF_KwargsWrapper,
    get_score_cutoff_f64, get_score_cutoff_i64,
    conv_sequence
)

from array import array
import numpy as np
cimport numpy as np

from numpy cimport npy_intp, PyArray_SimpleNew, PyArrayObject

from rapidfuzz_capi cimport (
    RF_Preprocess, RF_Kwargs, RF_String, RF_Scorer, RF_ScorerFunc,
    RF_Preprocessor, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64,
    RF_SCORER_FLAG_SYMMETRIC
)
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython.object cimport PyObject

np.import_array()

cdef extern from "cpp_process_cdist.hpp":
    object cdist_single_list_impl[T](  const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, int, int, T) except +
    object cdist_two_lists_impl[T](    const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, const vector[RF_StringWrapper]&, int, int, T) except +

    void set_score(PyArrayObject*, int, npy_intp, npy_intp, double)

cdef inline vector[PyObjectWrapper] preprocess_py(queries, processor) except *:
    cdef vector[PyObjectWrapper] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    proc_queries.reserve(queries_len)

    # processor None/False
    if not processor:
        for query in queries:
            proc_queries.emplace_back(<PyObject*>query)
    # processor has to be called through python
    else:
        for query in queries:
            proc_query = processor(query)
            proc_queries.emplace_back(<PyObject*>proc_query)

    return move(proc_queries)

cdef inline vector[RF_StringWrapper] preprocess(queries, processor) except *:
    cdef vector[RF_StringWrapper] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    cdef RF_String proc_str
    cdef RF_Preprocessor* processor_context = NULL
    proc_queries.reserve(queries_len)

    # No processor
    if not processor:
        for query in queries:
            proc_queries.emplace_back(conv_sequence(query))
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        # use RapidFuzz C-Api
        if processor_context != NULL and processor_context.version == 1:
            for query in queries:
                processor_context.preprocess(query, &proc_str)
                proc_queries.emplace_back(proc_str)

        # Call Processor through Python
        else:
            for query in queries:
                proc_query = processor(query)
                proc_queries.emplace_back(conv_sequence(proc_query), <PyObject*>proc_query)

    return move(proc_queries)

cdef inline int dtype_to_type_num(dtype) except -1:
    if dtype is np.int32:
        return np.NPY_INT32
    if dtype is np.int8:
        return np.NPY_INT8
    if dtype is np.int16:
        return np.NPY_INT16
    if dtype is np.int64:
        return np.NPY_INT64
    if dtype is np.uint8:
        return np.NPY_UINT8
    if dtype is np.uint16:
        return np.NPY_UINT16
    if dtype is np.uint32:
        return np.NPY_UINT32
    if dtype is np.uint64:
        return np.NPY_UINT64
    if dtype is np.float32:
        return np.NPY_FLOAT32
    if dtype is np.float64:
        return np.NPY_FLOAT64

    raise TypeError("unsupported dtype")

cdef inline int dtype_to_type_num_f64(dtype) except -1:
    if dtype is None:
        return np.NPY_FLOAT32
    return dtype_to_type_num(dtype)

cdef inline int dtype_to_type_num_i64(dtype) except -1:
    if dtype is None:
        return np.NPY_INT32
    return dtype_to_type_num(dtype)

cdef cdist_two_lists(queries, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    proc_choices = preprocess(choices, processor)
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))

    raise ValueError("scorer does not properly use the C-API")

cdef cdist_single_list(queries, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))

    raise ValueError("scorer does not properly use the C-API")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cdist_py(queries, choices, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
    proc_queries = preprocess_py(queries, processor)
    proc_choices = preprocess_py(choices, processor)

    cdef npy_intp[2] dims = [<npy_intp>proc_queries.size(), <npy_intp>proc_choices.size()]

    c_dtype = dtype_to_type_num_f64(dtype)
    matrix = PyArray_SimpleNew(2, dims, c_dtype)

    if score_cutoff is None:
        score_cutoff = 0
    elif score_cutoff < 0 or score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    for i in range(proc_queries.size()):
        for j in range(proc_choices.size()):
            score = scorer(<object>proc_queries[i].obj, <object>proc_choices[j].obj,**kwargs)
            set_score(<PyArrayObject*>matrix, c_dtype, i, j, score)

    return matrix

# todo link to c api docs
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

    cdef RF_Scorer* scorer_context = NULL
    cdef RF_ScorerFlags scorer_flags

    if processor is True:
        # todo: deprecate this
        processor = default_process
    elif processor is False:
        processor = None

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    if scorer_context:
        if dereference(scorer_context).version == 1:
            kwargs_context = RF_KwargsWrapper()
            dereference(scorer_context).kwargs_init(&kwargs_context.kwargs, kwargs)
            dereference(scorer_context).get_scorer_flags(&kwargs_context.kwargs, &scorer_flags)

            # scorer(queries[i], choices[j]) == scorer(queries[j], choices[i])
            if scorer_flags.flags & RF_SCORER_FLAG_SYMMETRIC and queries is choices:
                return cdist_single_list(
                    queries, scorer_context, &scorer_flags, processor,
                    score_cutoff, dtype, workers, &kwargs_context.kwargs)
            else:
                return cdist_two_lists(
                    queries, choices, scorer_context, &scorer_flags, processor,
                    score_cutoff, dtype, workers, &kwargs_context.kwargs)

    return cdist_py(queries, choices, scorer, processor, score_cutoff, dtype, workers, kwargs)
