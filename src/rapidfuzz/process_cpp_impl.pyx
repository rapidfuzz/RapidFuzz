# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.utils import default_process
from rapidfuzz.fuzz import WRatio, ratio

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport algorithm
from libcpp.utility cimport move
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t
from libc.math cimport floor

from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
cimport cython
from cython.operator cimport dereference
from cpython.exc cimport PyErr_CheckSignals
from cpython cimport Py_buffer
from cpython.buffer cimport PyBUF_ND, PyBUF_SIMPLE, PyBUF_F_CONTIGUOUS
from cpython.object cimport PyObject

from cpp_common cimport (
    PyObjectWrapper, RF_StringWrapper, RF_KwargsWrapper,
    conv_sequence, get_score_cutoff_f64, get_score_cutoff_i64
)

import heapq
from array import array

from rapidfuzz_capi cimport (
    RF_Preprocess, RF_Kwargs, RF_String, RF_Scorer, RF_ScorerFunc,
    RF_Preprocessor, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64,
    RF_SCORER_FLAG_SYMMETRIC
)
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

cdef extern from "process_cpp.hpp":
    cdef cppclass ExtractComp:
        ExtractComp()
        ExtractComp(const RF_ScorerFlags* scorer_flags)

    cdef cppclass ListMatchElem[T]:
        T score
        int64_t index
        PyObjectWrapper choice

    cdef cppclass DictMatchElem[T]:
        T score
        int64_t index
        PyObjectWrapper choice
        PyObjectWrapper key

    cdef cppclass DictStringElem:
        DictStringElem()
        DictStringElem(int64_t index, PyObjectWrapper key, PyObjectWrapper val, RF_StringWrapper proc_val)

        int64_t index
        PyObjectWrapper key
        PyObjectWrapper val
        RF_StringWrapper proc_val

    cdef cppclass ListStringElem:
        ListStringElem()
        ListStringElem(int64_t index, PyObjectWrapper val, RF_StringWrapper proc_val)

        int64_t index
        PyObjectWrapper val
        RF_StringWrapper proc_val

    cdef cppclass RF_ScorerWrapper:
        RF_ScorerFunc scorer_func

        RF_ScorerWrapper()
        RF_ScorerWrapper(RF_ScorerFunc)

        void call(const RF_String*, double, double*) except +
        void call(const RF_String*, int64_t, int64_t*) except +

    cdef vector[DictMatchElem[T]] extract_dict_impl[T](
        const RF_Kwargs*, const RF_ScorerFlags*, RF_Scorer*,
        const RF_StringWrapper&, const vector[DictStringElem]&, T) except +

    cdef vector[ListMatchElem[T]] extract_list_impl[T](
        const RF_Kwargs*, const RF_ScorerFlags*, RF_Scorer*,
        const RF_StringWrapper&, const vector[ListStringElem]&, T) except +

    cdef bool is_lowest_score_worst[T](const RF_ScorerFlags* scorer_flags)
    cdef T get_optimal_score[T](const RF_ScorerFlags* scorer_flags)

    cpdef enum class MatrixType:
        UNDEFINED = 0
        FLOAT32 = 1
        FLOAT64 = 2
        INT8 = 3
        INT16 = 4
        INT32 = 5
        INT64 = 6
        UINT8 = 7
        UINT16 = 8
        UINT32 = 9
        UINT64 = 10

    cdef cppclass RfMatrix "Matrix":
        RfMatrix() except +
        RfMatrix(MatrixType, size_t, size_t) except +
        int get_dtype_size() except +
        const char* get_format() except +
        void set[T](size_t, size_t, T) except +

        MatrixType m_dtype
        size_t m_rows
        size_t m_cols
        void* m_matrix

    RfMatrix cdist_single_list_impl[T](  const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, MatrixType, int, T) except +
    RfMatrix cdist_two_lists_impl[T](    const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, const vector[RF_StringWrapper]&, MatrixType, int, T) except +

cdef inline vector[DictStringElem] preprocess_dict(queries, processor) except *:
    cdef vector[DictStringElem] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    cdef RF_String proc_str
    cdef RF_Preprocessor* processor_context = NULL
    proc_queries.reserve(queries_len)
    cdef int64_t i

    # No processor
    if not processor:
        for i, (query_key, query) in enumerate(queries.items()):
            if query is None:
                continue
            proc_queries.emplace_back(
                i,
                move(PyObjectWrapper(query_key)),
                move(PyObjectWrapper(query)),
                move(RF_StringWrapper(conv_sequence(query)))
            )
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        # use RapidFuzz C-Api
        if processor_context != NULL and processor_context.version == 1:
            for i, (query_key, query) in enumerate(queries.items()):
                if query is None:
                    continue
                processor_context.preprocess(query, &proc_str)
                proc_queries.emplace_back(
                    i,
                    move(PyObjectWrapper(query_key)),
                    move(PyObjectWrapper(query)),
                    move(RF_StringWrapper(proc_str))
                )

        # Call Processor through Python
        else:
            for i, (query_key, query) in enumerate(queries.items()):
                if query is None:
                    continue
                proc_query = processor(query)
                proc_queries.emplace_back(
                    i,
                    move(PyObjectWrapper(query_key)),
                    move(PyObjectWrapper(query)),
                    move(RF_StringWrapper(conv_sequence(proc_query), proc_query))
                )

    return move(proc_queries)

cdef inline vector[ListStringElem] preprocess_list(queries, processor) except *:
    cdef vector[ListStringElem] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    cdef RF_String proc_str
    cdef RF_Preprocessor* processor_context = NULL
    proc_queries.reserve(queries_len)
    cdef int64_t i

    # No processor
    if not processor:
        for i, query in enumerate(queries):
            if query is None:
                continue
            proc_queries.emplace_back(
                i,
                move(PyObjectWrapper(query)),
                move(RF_StringWrapper(conv_sequence(query)))
            )
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        # use RapidFuzz C-Api
        if processor_context != NULL and processor_context.version == 1:
            for i, query in enumerate(queries):
                if query is None:
                    continue
                processor_context.preprocess(query, &proc_str)
                proc_queries.emplace_back(
                    i,
                    move(PyObjectWrapper(query)),
                    move(RF_StringWrapper(proc_str))
                )

        # Call Processor through Python
        else:
            for i, query in enumerate(queries):
                if query is None:
                    continue
                proc_query = processor(query)
                proc_queries.emplace_back(
                    i,
                    move(PyObjectWrapper(query)),
                    move(RF_StringWrapper(conv_sequence(proc_query), proc_query))
                )

    return move(proc_queries)

cdef inline extractOne_dict_f64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    cdef RF_String proc_str
    cdef double score
    cdef Py_ssize_t i = 0
    cdef RF_Preprocessor* processor_context = NULL
    if processor:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

    cdef RF_StringWrapper proc_query = move(RF_StringWrapper(conv_sequence(query)))
    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, scorer_flags)

    cdef RF_ScorerFunc scorer_func
    dereference(scorer).scorer_func_init(&scorer_func, kwargs, 1, &proc_query.string)
    cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)

    cdef bool lowest_score_worst = is_lowest_score_worst[double](scorer_flags)
    cdef double optimal_score = get_optimal_score[double](scorer_flags)

    cdef bool result_found = False
    cdef double result_score = 0
    result_key = None
    result_choice = None

    for choice_key, choice in choices.items():
        if i % 1000 == 0:
            PyErr_CheckSignals()
        i += 1
        if choice is None:
            continue

        if processor is None:
            proc_choice = move(RF_StringWrapper(conv_sequence(choice)))
        elif processor_context != NULL and processor_context.version == 1:
            processor_context.preprocess(choice, &proc_str)
            proc_choice = move(RF_StringWrapper(proc_str))
        else:
            py_proc_choice = processor(choice)
            proc_choice = move(RF_StringWrapper(conv_sequence(py_proc_choice)))

        ScorerFunc.call(&proc_choice.string, c_score_cutoff, &score)

        if lowest_score_worst:
            if score >= c_score_cutoff and (not result_found or score > result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_key = choice_key
                result_found = True
        else:
            if score <= c_score_cutoff and (not result_found or score < result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_key = choice_key
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_key) if result_found else None


cdef inline extractOne_dict_i64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    cdef RF_String proc_str
    cdef int64_t score
    cdef Py_ssize_t i = 0
    cdef RF_Preprocessor* processor_context = NULL
    if processor:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

    cdef RF_StringWrapper proc_query = move(RF_StringWrapper(conv_sequence(query)))
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, scorer_flags)

    cdef RF_ScorerFunc scorer_func
    dereference(scorer).scorer_func_init(&scorer_func, kwargs, 1, &proc_query.string)
    cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)

    cdef bool lowest_score_worst = is_lowest_score_worst[int64_t](scorer_flags)
    cdef int64_t optimal_score = get_optimal_score[int64_t](scorer_flags)

    cdef bool result_found = False
    cdef int64_t result_score = 0
    result_key = None
    result_choice = None

    for choice_key, choice in choices.items():
        if i % 1000 == 0:
            PyErr_CheckSignals()
        i += 1
        if choice is None:
            continue

        if processor is None:
            proc_choice = move(RF_StringWrapper(conv_sequence(choice)))
        elif processor_context != NULL and processor_context.version == 1:
            processor_context.preprocess(choice, &proc_str)
            proc_choice = move(RF_StringWrapper(proc_str))
        else:
            py_proc_choice = processor(choice)
            proc_choice = move(RF_StringWrapper(conv_sequence(py_proc_choice)))

        ScorerFunc.call(&proc_choice.string, c_score_cutoff, &score)

        if lowest_score_worst:
            if score >= c_score_cutoff and (not result_found or score > result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_key = choice_key
                result_found = True
        else:
            if score <= c_score_cutoff and (not result_found or score < result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_key = choice_key
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_key) if result_found else None


cdef inline extractOne_dict(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return extractOne_dict_f64(
            query, choices, scorer, scorer_flags, processor, score_cutoff, kwargs
        )
    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return extractOne_dict_i64(
            query, choices, scorer, scorer_flags, processor, score_cutoff, kwargs
        )

    raise ValueError("scorer does not properly use the C-API")

cdef inline extractOne_list_f64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    cdef RF_String proc_str
    cdef double score
    cdef Py_ssize_t i
    cdef RF_Preprocessor* processor_context = NULL
    if processor:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

    cdef RF_StringWrapper proc_query = move(RF_StringWrapper(conv_sequence(query)))
    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, scorer_flags)

    cdef RF_ScorerFunc scorer_func
    dereference(scorer).scorer_func_init(&scorer_func, kwargs, 1, &proc_query.string)
    cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)

    cdef bool lowest_score_worst = is_lowest_score_worst[double](scorer_flags)
    cdef double optimal_score = get_optimal_score[double](scorer_flags)

    cdef bool result_found = False
    cdef double result_score = 0
    cdef Py_ssize_t result_index = 0
    result_choice = None

    for i, choice in enumerate(choices):
        if i % 1000 == 0:
            PyErr_CheckSignals()
        if choice is None:
            continue

        if processor is None:
            proc_choice = move(RF_StringWrapper(conv_sequence(choice)))
        elif processor_context != NULL and processor_context.version == 1:
            processor_context.preprocess(choice, &proc_str)
            proc_choice = move(RF_StringWrapper(proc_str))
        else:
            py_proc_choice = processor(choice)
            proc_choice = move(RF_StringWrapper(conv_sequence(py_proc_choice)))

        ScorerFunc.call(&proc_choice.string, c_score_cutoff, &score)

        if lowest_score_worst:
            if score >= c_score_cutoff and (not result_found or score > result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_index = i
                result_found = True
        else:
            if score <= c_score_cutoff and (not result_found or score < result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_index = i
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_index) if result_found else None

cdef inline extractOne_list_i64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    cdef RF_String proc_str
    cdef int64_t score
    cdef Py_ssize_t i
    cdef RF_Preprocessor* processor_context = NULL
    if processor:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

    cdef RF_StringWrapper proc_query = move(RF_StringWrapper(conv_sequence(query)))
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, scorer_flags)

    cdef RF_ScorerFunc scorer_func
    dereference(scorer).scorer_func_init(&scorer_func, kwargs, 1, &proc_query.string)
    cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)

    cdef bool lowest_score_worst = is_lowest_score_worst[int64_t](scorer_flags)
    cdef int64_t optimal_score = get_optimal_score[int64_t](scorer_flags)

    cdef bool result_found = False
    cdef int64_t result_score = 0
    cdef Py_ssize_t result_index = 0
    result_choice = None

    for i, choice in enumerate(choices):
        if i % 1000 == 0:
            PyErr_CheckSignals()
        if choice is None:
            continue

        if processor is None:
            proc_choice = move(RF_StringWrapper(conv_sequence(choice)))
        elif processor_context != NULL and processor_context.version == 1:
            processor_context.preprocess(choice, &proc_str)
            proc_choice = move(RF_StringWrapper(proc_str))
        else:
            py_proc_choice = processor(choice)
            proc_choice = move(RF_StringWrapper(conv_sequence(py_proc_choice)))

        ScorerFunc.call(&proc_choice.string, c_score_cutoff, &score)

        if lowest_score_worst:
            if score >= c_score_cutoff and (not result_found or score > result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_index = i
                result_found = True
        else:
            if score <= c_score_cutoff and (not result_found or score < result_score):
                result_score = c_score_cutoff = score
                result_choice = choice
                result_index = i
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_index) if result_found else None

cdef inline extractOne_list(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, const RF_Kwargs* kwargs):
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return extractOne_list_f64(
            query, choices, scorer, scorer_flags, processor, score_cutoff, kwargs
        )
    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return extractOne_list_i64(
            query, choices, scorer, scorer_flags, processor, score_cutoff, kwargs
        )

    raise ValueError("scorer does not properly use the C-API")


cdef inline get_scorer_flags_py(scorer, dict kwargs):
    params = getattr(scorer, '_RF_ScorerPy', None)
    if params is not None:
        flags = params["get_scorer_flags"](**kwargs)
        return (flags["worst_score"], flags["optimal_score"])
    return (0, 100)

cdef inline py_extractOne_dict(query, choices, scorer, processor, double score_cutoff, worst_score, optimal_score, dict kwargs):
    cdef bool lowest_score_worst = optimal_score > worst_score
    cdef bool result_found = False
    result_score = 0
    result_choice = None
    result_key = None

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if lowest_score_worst:
            if score >= score_cutoff and (not result_found or score > result_score):
                score_cutoff = score
                kwargs["score_cutoff"] = score
                result_score = score
                result_choice = choice
                result_key = choice_key
                result_found = True
        else:
            if score <= score_cutoff and (not result_found or score < result_score):
                score_cutoff = score
                kwargs["score_cutoff"] = score
                result_score = score
                result_choice = choice
                result_key = choice_key
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_key) if result_choice is not None else None


cdef inline py_extractOne_list(query, choices, scorer, processor, double score_cutoff, worst_score, optimal_score, dict kwargs):
    cdef bool lowest_score_worst = optimal_score > worst_score
    cdef bool result_found = False
    cdef int64_t result_index = 0
    cdef int64_t i
    result_score = 0
    result_choice = None

    for i, choice in enumerate(choices):
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if lowest_score_worst:
            if score >= score_cutoff and (not result_found or score > result_score):
                score_cutoff = score
                kwargs["score_cutoff"] = score
                result_score = score
                result_choice = choice
                result_index = i
                result_found = True
        else:
            if score <= score_cutoff and (not result_found or score < result_score):
                score_cutoff = score
                kwargs["score_cutoff"] = score
                result_score = score
                result_choice = choice
                result_index = i
                result_found = True

        if score == optimal_score:
            break

    return (result_choice, result_score, result_index) if result_choice is not None else None


def extractOne(query, choices, *, scorer=WRatio, processor=default_process, score_cutoff=None, **kwargs):
    cdef RF_Scorer* scorer_context = NULL
    cdef RF_ScorerFlags scorer_flags

    if query is None:
        return None

    if processor is True:
        # todo: deprecate this
        processor = default_process
    elif processor is False:
        processor = None

    # preprocess the query
    if callable(processor):
        query = processor(query)


    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    if scorer_context and dereference(scorer_context).version == 1:
        kwargs_context = RF_KwargsWrapper()
        dereference(scorer_context).kwargs_init(&kwargs_context.kwargs, kwargs)
        dereference(scorer_context).get_scorer_flags(&kwargs_context.kwargs, &scorer_flags)

        if hasattr(choices, "items"):
            return extractOne_dict(query, choices, scorer_context, &scorer_flags,
                processor, score_cutoff, &kwargs_context.kwargs)
        else:
            return extractOne_list(query, choices, scorer_context, &scorer_flags,
                processor, score_cutoff, &kwargs_context.kwargs)


    worst_score, optimal_score = get_scorer_flags_py(scorer, kwargs)
    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = worst_score

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        return py_extractOne_dict(query, choices, scorer, processor, score_cutoff, worst_score, optimal_score, kwargs)
    else:
        return py_extractOne_list(query, choices, scorer, processor, score_cutoff, worst_score, optimal_score, kwargs)


cdef inline extract_dict_f64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    proc_query = move(RF_StringWrapper(conv_sequence(query)))
    proc_choices = preprocess_dict(choices, processor)

    cdef vector[DictMatchElem[double]] results = extract_dict_impl[double](
        kwargs, scorer_flags, scorer, proc_query, proc_choices,
        get_score_cutoff_f64(score_cutoff, scorer_flags))

    # due to score_cutoff not always completely filled
    if limit > <int64_t>results.size():
        limit = <int64_t>results.size()

    if limit >= <int64_t>results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp(scorer_flags))
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp(scorer_flags))
        results.resize(limit)

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        result_item = (<object>results[i].choice.obj, results[i].score, <object>results[i].key.obj)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list


cdef inline extract_dict_i64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    proc_query = move(RF_StringWrapper(conv_sequence(query)))
    proc_choices = preprocess_dict(choices, processor)

    cdef vector[DictMatchElem[int64_t]] results = extract_dict_impl[int64_t](
        kwargs, scorer_flags, scorer, proc_query, proc_choices,
        get_score_cutoff_i64(score_cutoff, scorer_flags))

    # due to score_cutoff not always completely filled
    if limit > <int64_t>results.size():
        limit = <int64_t>results.size()

    if limit >= <int64_t>results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp(scorer_flags))
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp(scorer_flags))
        results.resize(limit)

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        result_item = (<object>results[i].choice.obj, results[i].score, <object>results[i].key.obj)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list


cdef inline extract_dict(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return extract_dict_f64(
            query, choices, scorer, scorer_flags, processor, limit, score_cutoff, kwargs
        )
    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return extract_dict_i64(
            query, choices, scorer, scorer_flags, processor, limit, score_cutoff, kwargs
        )

    raise ValueError("scorer does not properly use the C-API")


cdef inline extract_list_f64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    proc_query = move(RF_StringWrapper(conv_sequence(query)))
    proc_choices = preprocess_list(choices, processor)

    cdef vector[ListMatchElem[double]] results = extract_list_impl[double](
        kwargs, scorer_flags, scorer, proc_query, proc_choices,
        get_score_cutoff_f64(score_cutoff, scorer_flags))

    # due to score_cutoff not always completely filled
    if limit > <int64_t>results.size():
        limit = <int64_t>results.size()

    if limit >= <int64_t>results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp(scorer_flags))
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp(scorer_flags))
        results.resize(limit)

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        result_item = (<object>results[i].choice.obj, results[i].score, results[i].index)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list


cdef inline extract_list_i64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    proc_query = move(RF_StringWrapper(conv_sequence(query)))
    proc_choices = preprocess_list(choices, processor)

    cdef vector[ListMatchElem[int64_t]] results = extract_list_impl[int64_t](
        kwargs, scorer_flags, scorer, proc_query, proc_choices,
        get_score_cutoff_i64(score_cutoff, scorer_flags))

    # due to score_cutoff not always completely filled
    if limit > <int64_t>results.size():
        limit = <int64_t>results.size()

    if limit >= <int64_t>results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp(scorer_flags))
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp(scorer_flags))
        results.resize(limit)

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        result_item = (<object>results[i].choice.obj, results[i].score, results[i].index)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list


cdef inline extract_list(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    flags = dereference(scorer_flags).flags

    if flags & RF_SCORER_FLAG_RESULT_F64:
        return extract_list_f64(
            query, choices, scorer, scorer_flags, processor, limit, score_cutoff, kwargs
        )
    elif flags & RF_SCORER_FLAG_RESULT_I64:
        return extract_list_i64(
            query, choices, scorer, scorer_flags, processor, limit, score_cutoff, kwargs
        )

    raise ValueError("scorer does not properly use the C-API")


cdef inline py_extract_dict(query, choices, scorer, processor, int64_t limit, double score_cutoff, worst_score, optimal_score, dict kwargs):
    cdef bool lowest_score_worst = optimal_score > worst_score
    cdef object score = None
    cdef list result_list = []

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if lowest_score_worst:
            if score >= score_cutoff:
                result_list.append((choice, score, choice_key))
        else:
            if score <= score_cutoff:
                result_list.append((choice, score, choice_key))

    if lowest_score_worst:
        return heapq.nlargest(limit, result_list, key=lambda i: i[1])
    else:
        return heapq.nsmallest(limit, result_list, key=lambda i: i[1])


cdef inline py_extract_list(query, choices, scorer, processor, int64_t limit, double score_cutoff, worst_score, optimal_score, dict kwargs):
    cdef bool lowest_score_worst = optimal_score > worst_score
    cdef object score = None
    cdef list result_list = []
    cdef int64_t i

    for i, choice in enumerate(choices):
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if lowest_score_worst:
            if score >= score_cutoff:
                result_list.append((choice, score, i))
        else:
            if score <= score_cutoff:
                result_list.append((choice, score, i))

    if lowest_score_worst:
        return heapq.nlargest(limit, result_list, key=lambda i: i[1])
    else:
        return heapq.nsmallest(limit, result_list, key=lambda i: i[1])


def extract(query, choices, *, scorer=WRatio, processor=default_process, limit=5, score_cutoff=None, **kwargs):
    cdef RF_Scorer* scorer_context = NULL
    cdef RF_ScorerFlags scorer_flags

    if query is None:
        return []

    if processor is True:
        processor = default_process
    elif processor is False:
        processor = None

    if limit is None or limit > len(choices):
        limit = len(choices)

    # preprocess the query
    if callable(processor):
        query = processor(query)

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    if scorer_context and dereference(scorer_context).version == 1:
        kwargs_context = RF_KwargsWrapper()
        dereference(scorer_context).kwargs_init(&kwargs_context.kwargs, kwargs)
        dereference(scorer_context).get_scorer_flags(&kwargs_context.kwargs, &scorer_flags)

        if hasattr(choices, "items"):
            return extract_dict(query, choices, scorer_context, &scorer_flags,
                processor, limit, score_cutoff, &kwargs_context.kwargs)
        else:
            return extract_list(query, choices, scorer_context, &scorer_flags,
                processor, limit, score_cutoff, &kwargs_context.kwargs)


    worst_score, optimal_score = get_scorer_flags_py(scorer, kwargs)
    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = worst_score

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        return py_extract_dict(query, choices, scorer, processor, limit, score_cutoff, worst_score, optimal_score, kwargs)
    else:
        return py_extract_list(query, choices, scorer, processor, limit, score_cutoff, worst_score, optimal_score, kwargs)


def extract_iter(query, choices, *, scorer=WRatio, processor=default_process, score_cutoff=None, **kwargs):
    cdef RF_Scorer* scorer_context = NULL
    cdef RF_ScorerFlags scorer_flags
    cdef RF_Preprocessor* processor_context = NULL
    cdef RF_KwargsWrapper kwargs_context

    def extract_iter_dict_f64():
        """
        implementation of extract_iter for dict, scorer using RapidFuzz C-API with the result type
        float64
        """
        cdef RF_String proc_str
        cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, &scorer_flags)
        query_proc = RF_StringWrapper(conv_sequence(query))

        cdef RF_ScorerFunc scorer_func
        dereference(scorer_context).scorer_func_init(
            &scorer_func, &kwargs_context.kwargs, 1, &query_proc.string
        )
        cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)
        cdef bool lowest_score_worst = is_lowest_score_worst[double](&scorer_flags)
        cdef double score

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            # use RapidFuzz C-Api
            if processor_context != NULL and processor_context.version == 1:
                processor_context.preprocess(choice, &proc_str)
                choice_proc = RF_StringWrapper(proc_str)
            elif processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                choice_proc = RF_StringWrapper(conv_sequence(proc_choice))
            else:
                choice_proc = RF_StringWrapper(conv_sequence(choice))

            ScorerFunc.call(&choice_proc.string, c_score_cutoff, &score)
            if lowest_score_worst:
                if score >= c_score_cutoff:
                    yield (choice, score, choice_key)
            else:
                if score <= c_score_cutoff:
                    yield (choice, score, choice_key)

    def extract_iter_dict_i64():
        """
        implementation of extract_iter for dict, scorer using RapidFuzz C-API with the result type
        int64_t
        """
        cdef RF_String proc_str
        cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, &scorer_flags)
        query_proc = RF_StringWrapper(conv_sequence(query))

        cdef RF_ScorerFunc scorer_func
        dereference(scorer_context).scorer_func_init(
            &scorer_func, &kwargs_context.kwargs, 1, &query_proc.string
        )
        cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)
        cdef bool lowest_score_worst = is_lowest_score_worst[int64_t](&scorer_flags)
        cdef int64_t score

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            # use RapidFuzz C-Api
            if processor_context != NULL and processor_context.version == 1:
                processor_context.preprocess(choice, &proc_str)
                choice_proc = RF_StringWrapper(proc_str)
            elif processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                choice_proc = RF_StringWrapper(conv_sequence(proc_choice))
            else:
                choice_proc = RF_StringWrapper(conv_sequence(choice))

            ScorerFunc.call(&choice_proc.string, c_score_cutoff, &score)
            if lowest_score_worst:
                if score >= c_score_cutoff:
                    yield (choice, score, choice_key)
            else:
                if score <= c_score_cutoff:
                    yield (choice, score, choice_key)

    def extract_iter_list_f64():
        """
        implementation of extract_iter for list, scorer using RapidFuzz C-API with the result type
        float64
        """
        cdef RF_String proc_str
        cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, &scorer_flags)
        query_proc = RF_StringWrapper(conv_sequence(query))

        cdef RF_ScorerFunc scorer_func
        dereference(scorer_context).scorer_func_init(
            &scorer_func, &kwargs_context.kwargs, 1, &query_proc.string
        )
        cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)
        cdef bool lowest_score_worst = is_lowest_score_worst[double](&scorer_flags)
        cdef double score

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            # use RapidFuzz C-Api
            if processor_context != NULL and processor_context.version == 1:
                processor_context.preprocess(choice, &proc_str)
                choice_proc = RF_StringWrapper(proc_str)
            elif processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                choice_proc = RF_StringWrapper(conv_sequence(proc_choice))
            else:
                choice_proc = RF_StringWrapper(conv_sequence(choice))

            ScorerFunc.call(&choice_proc.string, c_score_cutoff, &score)
            if lowest_score_worst:
                if score >= c_score_cutoff:
                    yield (choice, score, i)
            else:
                if score <= c_score_cutoff:
                    yield (choice, score, i)

    def extract_iter_list_i64():
        """
        implementation of extract_iter for list, scorer using RapidFuzz C-API with the result type
        int64_t
        """
        cdef RF_String proc_str
        cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, &scorer_flags)
        query_proc = RF_StringWrapper(conv_sequence(query))

        cdef RF_ScorerFunc scorer_func
        dereference(scorer_context).scorer_func_init(
            &scorer_func, &kwargs_context.kwargs, 1, &query_proc.string
        )
        cdef RF_ScorerWrapper ScorerFunc = RF_ScorerWrapper(scorer_func)
        cdef bool lowest_score_worst = is_lowest_score_worst[int64_t](&scorer_flags)
        cdef int64_t score

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            # use RapidFuzz C-Api
            if processor_context != NULL and processor_context.version == 1:
                processor_context.preprocess(choice, &proc_str)
                choice_proc = RF_StringWrapper(proc_str)
            elif processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                choice_proc = RF_StringWrapper(conv_sequence(proc_choice))
            else:
                choice_proc = RF_StringWrapper(conv_sequence(choice))

            ScorerFunc.call(&choice_proc.string, c_score_cutoff, &score)
            if lowest_score_worst:
                if score >= c_score_cutoff:
                    yield (choice, score, i)
            else:
                if score <= c_score_cutoff:
                    yield (choice, score, i)

    def py_extract_iter_dict(worst_score, optimal_score):
        """
        implementation of extract_iter for:
          - type of choices = dict
          - scorer = python function
        """
        cdef bool lowest_score_worst = optimal_score > worst_score

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            if processor is not None:
                score = scorer(query, processor(choice), **kwargs)
            else:
                score = scorer(query, choice, **kwargs)

            if lowest_score_worst:
                if score >= score_cutoff:
                    yield (choice, score, choice_key)
            else:
                if score <= score_cutoff:
                    yield (choice, score, choice_key)

    def py_extract_iter_list(worst_score, optimal_score):
        """
        implementation of extract_iter for:
          - type of choices = list
          - scorer = python function
        """
        cdef bool lowest_score_worst = optimal_score > worst_score
        cdef int64_t i

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                score = scorer(query, processor(choice), **kwargs)
            else:
                score = scorer(query, choice, **kwargs)

            if lowest_score_worst:
                if score >= score_cutoff:
                    yield (choice, score, i)
            else:
                if score <= score_cutoff:
                    yield (choice, score, i)

    if query is None:
        # finish generator
        return

    if processor is True:
        processor = default_process
    elif processor is False:
        processor = None

    # preprocess the query
    if callable(processor):
        query = processor(query)

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    if scorer_context and dereference(scorer_context).version == 1:
        kwargs_context = RF_KwargsWrapper()
        dereference(scorer_context).kwargs_init(&kwargs_context.kwargs, kwargs)
        dereference(scorer_context).get_scorer_flags(&kwargs_context.kwargs, &scorer_flags)

        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        if hasattr(choices, "items"):
            if scorer_flags.flags & RF_SCORER_FLAG_RESULT_F64:
                yield from extract_iter_dict_f64()
                return
            elif scorer_flags.flags & RF_SCORER_FLAG_RESULT_I64:
                yield from extract_iter_dict_i64()
                return
        else:
            if scorer_flags.flags & RF_SCORER_FLAG_RESULT_F64:
                yield from extract_iter_list_f64()
                return
            elif scorer_flags.flags & RF_SCORER_FLAG_RESULT_I64:
                yield from extract_iter_list_i64()
                return

        raise ValueError("scorer does not properly use the C-API")

    worst_score, optimal_score = get_scorer_flags_py(scorer, kwargs)
    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = worst_score

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        yield from py_extract_iter_dict(worst_score, optimal_score)
    else:
        yield from py_extract_iter_list(worst_score, optimal_score)


FLOAT32 = MatrixType.FLOAT32
FLOAT64 = MatrixType.FLOAT64
INT8 = MatrixType.INT8
INT16 = MatrixType.INT16
INT32 = MatrixType.INT32
INT64 = MatrixType.INT64
UINT8 = MatrixType.UINT8
UINT16 = MatrixType.UINT16
UINT32 = MatrixType.UINT32
UINT64 = MatrixType.UINT64

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

cdef inline MatrixType dtype_to_type_num_f64(dtype) except MatrixType.UNDEFINED:
    if dtype is None:
        return MatrixType.FLOAT32
    return <MatrixType>dtype

cdef inline MatrixType dtype_to_type_num_i64(dtype) except MatrixType.UNDEFINED:
    if dtype is None:
        return MatrixType.INT32
    return <MatrixType>dtype

from cpython cimport Py_buffer
from libcpp.vector cimport vector

cdef class Matrix:
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef RfMatrix matrix

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self.shape[0] = self.matrix.m_rows
        self.shape[1] = self.matrix.m_cols
        self.strides[1] = self.matrix.get_dtype_size()
        self.strides[0] = self.matrix.m_cols * self.strides[1]

        buffer.buf = <char *>self.matrix.m_matrix
        buffer.format = <char *>self.matrix.get_format()
        buffer.internal = NULL
        buffer.itemsize = self.matrix.get_dtype_size()
        buffer.len = self.matrix.m_rows * self.matrix.m_cols * self.matrix.get_dtype_size()
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef cdist_two_lists(queries, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    proc_choices = preprocess(choices, processor)
    flags = dereference(scorer_flags).flags
    cdef Matrix matrix = Matrix()

    if flags & RF_SCORER_FLAG_RESULT_F64:
        matrix.matrix = cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        matrix.matrix = cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))
    else:
        raise ValueError("scorer does not properly use the C-API")

    return matrix

cdef Matrix cdist_single_list(queries, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    flags = dereference(scorer_flags).flags
    cdef Matrix matrix = Matrix()

    if flags & RF_SCORER_FLAG_RESULT_F64:
        matrix.matrix = cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        matrix.matrix = cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))
    else:
        raise ValueError("scorer does not properly use the C-API")

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cdist_py(queries, choices, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
    proc_queries = preprocess_py(queries, processor)
    proc_choices = preprocess_py(choices, processor)
    cdef double score
    cdef Matrix matrix = Matrix()
    c_dtype = dtype_to_type_num_f64(dtype)
    matrix.matrix = RfMatrix(c_dtype, proc_queries.size(), proc_choices.size())

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    for i in range(proc_queries.size()):
        for j in range(proc_choices.size()):
            score = scorer(<object>proc_queries[i].obj, <object>proc_choices[j].obj,**kwargs)
            matrix.matrix.set(i, j, score)

    return matrix


def cdist(queries, choices, *, scorer=ratio, processor=None, score_cutoff=None, dtype=None, workers=1, **kwargs):
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
