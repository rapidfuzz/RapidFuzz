# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.utils import default_process
from rapidfuzz.fuzz import WRatio

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport algorithm
from libcpp.utility cimport move
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t
from libc.math cimport floor

from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference

from cpp_common cimport (
    PyObjectWrapper, RF_StringWrapper, RF_KwargsWrapper,
    conv_sequence, get_score_cutoff_f64, get_score_cutoff_i64
)

import heapq
from array import array

from rapidfuzz_capi cimport (
    RF_Kwargs, RF_String, RF_Scorer, RF_ScorerFunc,
    RF_Preprocessor, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64
)
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

cdef extern from "cpp_process.hpp":
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
    result_key = None
    result_choice = None

    for choice_key, choice in choices.items():
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
    result_key = None
    result_choice = None

    for choice_key, choice in choices.items():
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


cdef inline py_extractOne_dict(query, choices, scorer, processor, double score_cutoff, dict kwargs):
    result_score = -1
    result_choice = None
    result_key = None

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if score >= score_cutoff and score > result_score:
            score_cutoff = score
            kwargs["score_cutoff"] = score
            result_score = score
            result_choice = choice
            result_key = choice_key

            if score_cutoff == 100:
                break

    return (result_choice, result_score, result_key) if result_choice is not None else None


cdef inline py_extractOne_list(query, choices, scorer, processor, double score_cutoff, dict kwargs):
    cdef int64_t result_index = 0
    cdef int64_t i
    result_score = -1
    result_choice = None

    for i, choice in enumerate(choices):
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if score >= score_cutoff and score > result_score:
            score_cutoff = score
            kwargs["score_cutoff"] = score
            result_score = score
            result_choice = choice
            result_index = i

            if score_cutoff == 100:
                break

    return (result_choice, result_score, result_index) if result_choice is not None else None


def extractOne(query, choices, *, scorer=WRatio, processor=default_process, score_cutoff=None, **kwargs):
    """
    Find the best match in a list of choices. When multiple elements have the same similarity,
    the first element is returned.

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Iterable[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance <= score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity >= score_cutoff` are ignored. For edit distances this defaults to
        -1, while for normalized edit distances this defaults to 0.0, which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    Tuple[Sequence[Hashable], Any, Any]
        Returns the best match in form of a Tuple with 3 elements. The values stored in the
        tuple depend on the types of the input arguments.

        * The first element is always the `choice`, which is the value thats compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= score_cutoff` are returned.
            An example of a scorer with this behavior is `string_metric.levenshtein`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are returned.
            An example of a scorer with this behavior is `string_metric.normalized_levenshtein`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

    None
        When no choice has a `similarity >= score_cutoff`/`distance <= score_cutoff` None is returned

    Examples
    --------

    >>> from rapidfuzz.process import extractOne
    >>> from rapidfuzz.string_metric import levenshtein, normalized_levenshtein
    >>> from rapidfuzz.fuzz import ratio

    extractOne can be used with normalized edit distances.

    >>> extractOne("abcd", ["abce"], scorer=ratio)
    ("abcd", 75.0, 1)
    >>> extractOne("abcd", ["abce"], scorer=normalized_levenshtein)
    ("abcd", 75.0, 1)

    extractOne can be used with edit distances as well.

    >>> extractOne("abcd", ["abce"], scorer=levenshtein)
    ("abce", 1, 0)

    additional settings of the scorer can be passed as keyword arguments to extractOne

    >>> extractOne("abcd", ["abce"], scorer=levenshtein, weights=(1,1,2))
    ("abcde", 2, 1)

    when a mapping is used for the choices the key of the choice is returned instead of the List index

    >>> extractOne("abcd", {"key": "abce"}, scorer=ratio)
    ("abcd", 75.0, "key")

    By default each string is preprocessed using `utils.default_process`, which lowercases the strings,
    replaces non alphanumeric characters with whitespaces and trims whitespaces from start and end of them.
    This behavior can be changed by passing a custom function, or None to disable the behavior. Preprocessing
    can take a significant part of the runtime, so it makes sense to disable it, when it is not required.


    >>> extractOne("abcd", ["abdD"], scorer=ratio)
    ("abcD", 100.0, 0)
    >>> extractOne("abcd", ["abdD"], scorer=ratio, processor=None)
    ("abcD", 75.0, 0)
    >>> extractOne("abcd", ["abdD"], scorer=ratio, processor=lambda s: s.upper())
    ("abcD", 100.0, 0)

    When only results with a similarity above a certain threshold are relevant, the parameter score_cutoff can be
    used to filter out results with a lower similarity. This threshold is used by some of the scorers to exit early,
    when they are sure, that the similarity is below the threshold.
    For normalized edit distances all results with a similarity below score_cutoff are filtered out

    >>> extractOne("abcd", ["abce"], scorer=ratio)
    ("abce", 75.0, 0)
    >>> extractOne("abcd", ["abce"], scorer=ratio, score_cutoff=80)
    None

    For edit distances all results with an edit distance above the score_cutoff are filtered out

    >>> extractOne("abcd", ["abce"], scorer=levenshtein, weights=(1,1,2))
    ("abce", 2, 0)
    >>> extractOne("abcd", ["abce"], scorer=levenshtein, weights=(1,1,2), score_cutoff=1)
    None

    """
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


    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = 0
    elif score_cutoff < 0 or score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        return py_extractOne_dict(query, choices, scorer, processor, score_cutoff, kwargs)
    else:
        return py_extractOne_list(query, choices, scorer, processor, score_cutoff, kwargs)


cdef inline extract_dict_f64(query, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, int64_t limit, score_cutoff, const RF_Kwargs* kwargs):
    proc_query = move(RF_StringWrapper(conv_sequence(query)))
    proc_choices = preprocess_dict(choices, processor)

    cdef vector[DictMatchElem[double]] results = extract_dict_impl[double](
        kwargs, scorer_flags, scorer, proc_query, proc_choices,
        get_score_cutoff_f64(score_cutoff, scorer_flags))

    # due to score_cutoff not always completely filled
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
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
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
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
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
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
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
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


cdef inline py_extract_dict(query, choices, scorer, processor, int64_t limit, double score_cutoff, dict kwargs):
    cdef object score = None
    cdef list result_list = []

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            score = scorer(query, processor(choice), **kwargs)
        else:
            score = scorer(query, choice, **kwargs)

        if score >= score_cutoff:
            result_list.append((choice, score, choice_key))

    return heapq.nlargest(limit, result_list, key=lambda i: i[1])


cdef inline py_extract_list(query, choices, scorer, processor, int64_t limit, double score_cutoff, dict kwargs):
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

        if score >= score_cutoff:
            result_list.append((choice, score, i))

    return heapq.nlargest(limit, result_list, key=lambda i: i[1])


def extract(query, choices, *, scorer=WRatio, processor=default_process, limit=5, score_cutoff=None, **kwargs):
    """
    Find the best matches in a list of choices. The list is sorted by the similarity.
    When multiple choices have the same similarity, they are sorted by their index

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Collection[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    limit : int
        maximum amount of results to return
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance <= score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity >= score_cutoff` are ignored. For edit distances this defaults to
        -1, while for normalized edit distances this defaults to 0.0, which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    List[Tuple[Sequence[Hashable], Any, Any]]
        The return type is always a List of Tuples with 3 elements. However the values stored in the
        tuple depend on the types of the input arguments.

        * The first element is always the `choice`, which is the value thats compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= max` are returned.
            An example of a scorer with this behavior is `string_metric.levenshtein`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are returned.
            An example of a scorer with this behavior is `string_metric.normalized_levenshtein`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

        The list is sorted by `score_cutoff` or `max` depending on the scorer used. The first element in the list
        has the `highest similarity`/`smallest distance`.

    """
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


    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = 0
    elif score_cutoff < 0 or score_cutoff > 100:
        raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        return py_extract_dict(query, choices, scorer, processor, limit, score_cutoff, kwargs)
    else:
        return py_extract_list(query, choices, scorer, processor, limit, score_cutoff, kwargs)


def extract_iter(query, choices, *, scorer=WRatio, processor=default_process, score_cutoff=None, **kwargs):
    """
    Find the best match in a list of choices

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Iterable[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance <= score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity >= score_cutoff` are ignored. For edit distances this defaults to
        -1, while for normalized edit distances this defaults to 0.0, which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Yields
    -------
    Tuple[Sequence[Hashable], Any, Any]
        Yields similarity between the query and each choice in form of a Tuple with 3 elements.
        The values stored in the tuple depend on the types of the input arguments.

        * The first element is always the current `choice`, which is the value thats compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= max` are yielded.
            An example of a scorer with this behavior is `string_metric.levenshtein`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are yielded.
            An example of a scorer with this behavior is `string_metric.normalized_levenshtein`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

    """
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

    def py_extract_iter_dict():
        """
        implementation of extract_iter for:
          - type of choices = dict
          - scorer = python function
        """
        cdef double c_score_cutoff = score_cutoff

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            if processor is not None:
                score = scorer(query, processor(choice), **kwargs)
            else:
                score = scorer(query, choice, **kwargs)

            if score >= c_score_cutoff:
                yield (choice, score, choice_key)

    def py_extract_iter_list():
        """
        implementation of extract_iter for:
          - type of choices = list
          - scorer = python function
        """
        cdef int64_t i
        cdef double c_score_cutoff = score_cutoff

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                score = scorer(query, processor(choice), **kwargs)
            else:
                score = scorer(query, choice, **kwargs)

            if score >= c_score_cutoff:
                yield(choice, score, i)

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

    # the scorer has to be called through Python
    if score_cutoff is None:
        score_cutoff = 0.0

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    if hasattr(choices, "items"):
        yield from py_extract_iter_dict()
    else:
        yield from py_extract_iter_list()
