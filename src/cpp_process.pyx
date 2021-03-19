# distutils: language=c++
# cython: language_level=3
# cython: binding=True

from rapidfuzz.utils import default_process

from rapidfuzz.string_metric import (
    normalized_levenshtein,
    normalized_hamming
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
from libcpp cimport algorithm

from cpython.list cimport PyList_New
from cpython.list cimport PyList_SET_ITEM
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.ref cimport Py_DECREF

import heapq

cdef extern from "Python.h":
    # This isn't included in the cpython definitions
    # using PyObject* rather than object lets us control refcounting
    PyObject* Py_BuildValue(const char*,...) except NULL


cdef extern from "cpp_process.hpp":
    ctypedef double (*scorer_func) (void* context, object py_str, double score_cutoff) except +
    ctypedef void (*scorer_context_deinit) (void* context) except +

    ctypedef struct scorer_context:
        void* context
        scorer_func scorer
        scorer_context_deinit deinit

    # fuzz
    scorer_context cached_ratio_init(                   object, int) except +
    scorer_context cached_partial_ratio_init(           object, int) except +
    scorer_context cached_token_sort_ratio_init(        object, int) except +
    scorer_context cached_token_set_ratio_init(         object, int) except +
    scorer_context cached_token_ratio_init(             object, int) except +
    scorer_context cached_partial_token_sort_ratio_init(object, int) except +
    scorer_context cached_partial_token_set_ratio_init( object, int) except +
    scorer_context cached_partial_token_ratio_init(     object, int) except +
    scorer_context cached_WRatio_init(                  object, int) except +
    scorer_context cached_QRatio_init(                  object, int) except +
    # string_metric
    scorer_context cached_normalized_levenshtein_init(object, int, size_t, size_t, size_t) except +
    scorer_context cached_normalized_hamming_init(      object, int) except +



    ctypedef struct ExtractComp:
        pass

    ctypedef struct ListMatchElem:
        double score
        size_t index

    ctypedef struct DictMatchElem:
        double score
        size_t index
        PyObject* choice
        PyObject* key


cdef inline extractOne_dict(scorer_context context, choices, processor, double score_cutoff):
    cdef double score
    result_choice = None
    result_key = None

    if processor is not None:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = context.scorer(context.context, processor(choice), score_cutoff)

            if score >= score_cutoff:
                score_cutoff = score
                result_choice = choice
                result_key = choice_key

                if score_cutoff == 100:
                    break
    else:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = context.scorer(context.context, choice, score_cutoff)

            if score >= score_cutoff:
                score_cutoff = score
                result_choice = choice
                result_key = choice_key

                if score_cutoff == 100:
                    break

    return (result_choice, score_cutoff, result_key) if result_choice is not None else None


cdef inline extractOne_list(scorer_context context, choices, processor, double score_cutoff):
    cdef double score = 0.0
    cdef int index = 0
    cdef int result_index = 0
    result_choice = None

    if processor is not None:
        for choice in choices:
            if choice is None:
                continue

            score = context.scorer(context.context, processor(choice), score_cutoff)

            if score >= score_cutoff:
                score_cutoff = score
                result_choice = choice
                result_index = index

                if score_cutoff == 100:
                    break
            index += 1
    else:
        for choice in choices:
            if choice is None:
                continue

            score = context.scorer(context.context, choice, score_cutoff)

            if score >= score_cutoff:
                score_cutoff = score
                result_choice = choice
                result_index = index

                if score_cutoff == 100:
                    break
            index += 1

    return (result_choice, score_cutoff, result_index) if result_choice is not None else None


cdef inline py_extractOne_dict(query, choices, scorer, processor, double score_cutoff, kwargs):
    result_score = 0
    result_choice = None
    result_key = None

    if processor is not None:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = scorer(query, processor(choice),
                processor=None, score_cutoff=score_cutoff, **kwargs)

            if score > result_score:
                score_cutoff = score
                result_score = score
                result_choice = choice
                result_key = choice_key

                if score_cutoff == 100:
                    break
    else:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = scorer(query, choice,
                processor=None, score_cutoff=score_cutoff, **kwargs)

            if score > result_score:
                score_cutoff = score
                result_score = score
                result_choice = choice
                result_key = choice_key

                if score_cutoff == 100:
                    break

    return (result_choice, result_score, result_key) if result_choice is not None else None


cdef inline py_extractOne_list(query, choices, scorer, processor, double score_cutoff, kwargs):
    cdef int result_index = 0
    cdef int index = 0
    result_score = 0
    result_choice = None

    if processor is not None:
        for choice in choices:
            if choice is None:
                continue

            score = scorer(query, processor(choice),
                processor=None, score_cutoff=score_cutoff, **kwargs)

            if score > result_score:
                score_cutoff = score
                result_score = score
                result_choice = choice
                result_index = index

                if score_cutoff == 100:
                    break
            index += 1
    else:
        for choice in choices:
            if choice is None:
                continue

            score = scorer(query, choice,
                processor=None, score_cutoff=score_cutoff, **kwargs)

            if score > result_score:
                score_cutoff = score
                result_score = score
                result_choice = choice
                result_index = index

                if score_cutoff == 100:
                    break
            index += 1

    return (result_choice, result_score, result_index) if result_choice is not None else None


cdef inline scorer_context CachedNormalizedLevenshteinInit(object query, int def_process, dict kwargs):
    cdef size_t insertion, deletion, substitution
    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    return cached_normalized_levenshtein_init(query, def_process, insertion, deletion, substitution)


cdef inline scorer_context CachedScorerInit(object scorer, object query, int def_process, dict kwargs):
    cdef scorer_context context

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
    else:
        context.context = NULL
    return context


def extractOne(query, choices, scorer=WRatio, processor=default_process, double score_cutoff=0.0, **kwargs):
    """
    Find the best match in a list of choices

    Parameters
    ----------
    query : str
        string we want to find
    choices : Iterable
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. fuzz.WRatio is used by default
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        Matches with a lower score than this number will be ignored. Default is 0,
        which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    Union[None, Tuple[str, float, Any]]
        Returns the best match the best match
        in form of a tuple or None when there is no match with a score >= score_cutoff.
        The Tuple will be in the form `(<choice>, <ratio>, <index of choice>)`
        when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
        when `choices` is a mapping.
    """

    cdef int def_process = 0
    cdef scorer_context context

    if query is None:
        return None

    # preprocess the query
    if processor is default_process:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = processor(query)
        processor = None
    elif callable(processor):
        query = processor(query)
    elif processor:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = default_process(query)
        processor = None
    # query might be e.g. False
    else:
        processor = None

    # directly use the C++ implementation if possible
    context = CachedScorerInit(scorer, query, def_process, kwargs)
    if context.context != NULL:
        try:
            if hasattr(choices, "items"):
                return extractOne_dict(context, choices, processor, score_cutoff)
            else:
                return extractOne_list(context, choices, processor, score_cutoff)
        finally:
            # part of the context is dynamically allocated, so it has to be freed in any case
            context.deinit(context.context)
    # the scorer has to be called through Python
    else:
        if hasattr(choices, "items"):
            return py_extractOne_dict(query, choices, scorer, processor, score_cutoff, kwargs)
        else:
            return py_extractOne_list(query, choices, scorer, processor, score_cutoff, kwargs)


cdef inline extract_dict(scorer_context context, choices, processor, size_t limit, double score_cutoff):
    cdef double score = 0.0
    cdef size_t index = 0
    cdef size_t i = 0
    # todo storing 32 Byte per element is a bit wasteful
    # maybe store only key and access the corresponding element when building the list
    cdef vector[DictMatchElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    if processor is not None:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = context.scorer(context.context, processor(choice), score_cutoff)

            if score >= score_cutoff:
                # especially the key object might be created on the fly by e.g. pandas.Dataframe
                # so we need to ensure Python does not deallocate it
                Py_INCREF(choice)
                Py_INCREF(choice_key)
                results.push_back(DictMatchElem(score, i, <PyObject*>choice, <PyObject*>choice_key))
            index += 1
    else:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = context.scorer(context.context, choice, score_cutoff)

            if score >= score_cutoff:
                # especially the key object might be created on the fly by e.g. pandas.Dataframe
                # so we need to ensure Python does not deallocate it
                Py_INCREF(choice)
                Py_INCREF(choice_key)
                results.push_back(DictMatchElem(score, i, <PyObject*>choice, <PyObject*>choice_key))
            index += 1

    # due to score_cutoff not always completely filled
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp())
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp())

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        # PyList_SET_ITEM steals a reference
        # the casting is necessary to ensure that Cython doesn't
        # decref the result of Py_BuildValue
        # https://stackoverflow.com/questions/43553763/cythonize-list-of-all-splits-of-a-string/43557675#43557675
        PyList_SET_ITEM(result_list, <Py_ssize_t>i,
            <object>Py_BuildValue("OdO",
                <PyObject*>results[i].choice,
                results[i].score,
                <PyObject*>results[i].key))

    # decref all reference counts
    for i in range(results.size()):
        Py_DECREF(<object>results[i].choice)
        Py_DECREF(<object>results[i].key)

    return result_list


cdef inline extract_list(scorer_context context, choices, processor, size_t limit, double score_cutoff):
    cdef double score = 0.0
    cdef size_t index = 0
    cdef size_t i = 0
    # todo possibly a smaller vector would be good to reduce memory usage
    cdef vector[ListMatchElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    if processor is not None:
        for choice in choices:
            if choice is None:
                continue

            score = context.scorer(context.context, processor(choice), score_cutoff)

            if score >= score_cutoff:
                results.push_back(ListMatchElem(score, index))
            index += 1
    else:
        for choice in choices:
            if choice is None:
                continue

            score = context.scorer(context.context, choice, score_cutoff)

            if score >= score_cutoff:
                results.push_back(ListMatchElem(score, index))
            index += 1

    # due to score_cutoff not always completely filled
    if limit > results.size():
        limit = results.size()

    if limit >= results.size():
        algorithm.sort(results.begin(), results.end(), ExtractComp())
    else:
        algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractComp())

    # copy elements into Python List
    result_list = PyList_New(<Py_ssize_t>limit)
    for i in range(limit):
        # PyList_SET_ITEM steals a reference
        # the casting is necessary to ensure that Cython doesn't
        # decref the result of Py_BuildValue
        # https://stackoverflow.com/questions/43553763/cythonize-list-of-all-splits-of-a-string/43557675#43557675

        PyList_SET_ITEM(result_list, <Py_ssize_t>i,
            <object>Py_BuildValue("Odn",
                <PyObject*>choices[results[i].index],
                results[i].score,
                results[i].index))

    return result_list


cdef inline py_extract_dict(query, choices, scorer, processor, size_t limit, double score_cutoff, kwargs):
    cdef object score = None
    # todo working directly with a list is relatively slow
    # also it is not very memory efficient to allocate space for all elements even when only
    # a part is used. This should be optimised in the future
    cdef list result_list = []

    if processor is not None:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = scorer(query, processor(choice), score_cutoff, **kwargs)

            if score >= score_cutoff:
                result_list.append((choice, score, choice_key))
    else:
        for choice_key, choice in choices.items():
            if choice is None:
                continue

            score = scorer(query, choice, score_cutoff, **kwargs)

            if score >= score_cutoff:
                result_list.append((choice, score, choice_key))

    return heapq.nlargest(limit, result_list, key=lambda i: i[1])


cdef inline py_extract_list(query, choices, scorer, processor, size_t limit, double score_cutoff, kwargs):
    cdef object score = None
    # todo working directly with a list is relatively slow
    # also it is not very memory efficient to allocate space for all elements even when only
    # a part is used. This should be optimised in the future
    cdef list result_list = []
    cdef size_t index = 0

    if processor is not None:
        for choice in choices:
            if choice is None:
                continue

            score = scorer(query, processor(choice), score_cutoff, **kwargs)

            if score >= score_cutoff:
                result_list.append((choice, score, index))
            index += 1
    else:
        for choice in choices:
            if choice is None:
                continue

            score = scorer(query, choice, index, **kwargs)

            if score >= score_cutoff:
                result_list.append((choice, score, index))
            index += 1

    return heapq.nlargest(limit, result_list, key=lambda i: i[1])


def extract(query, choices, scorer=WRatio, processor=default_process, limit=5, double score_cutoff=0.0, **kwargs):
    """
    Find the best matches in a list of choices

    Parameters
    ----------
    query : str
        string we want to find
    choices : Iterable
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. fuzz.WRatio is used by default
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    limit : int
        maximum amount of results to return
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        Matches with a lower score than this number will be ignored. Default is 0,
        which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Returns
    -------
    List[Tuple[str, float, Any]]
        Returns a list of all matches that have a `score >= score_cutoff`.
        The list will be of either `(<choice>, <ratio>, <index of choice>)`
        when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
        when `choices` is a mapping

    """
    cdef int def_process = 0
    cdef scorer_context context

    if query is None:
        return None

    if limit is None or limit > len(choices):
        limit = len(choices)

    # preprocess the query
    if processor is default_process:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = processor(query)
        processor = None
    elif callable(processor):
        query = processor(query)
    elif processor:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = default_process(query)
        processor = None
    # query might be e.g. False
    else:
        processor = None

    # directly use the C++ implementation if possible
    context = CachedScorerInit(scorer, query, def_process, kwargs)
    if context.context != NULL:
        try:
            if hasattr(choices, "items"):
                return extract_dict(context, choices, processor, limit, score_cutoff)
            else:
                return extract_list(context, choices, processor, limit, score_cutoff)

        finally:
            # part of the context is dynamically allocated, so it has to be freed in any case
            context.deinit(context.context)
    # the scorer has to be called through Python
    else:
        if hasattr(choices, "items"):
            return py_extract_dict(query, choices, scorer, processor, limit, score_cutoff, kwargs)
        else:
            return py_extract_list(query, choices, scorer, processor, limit, score_cutoff, kwargs)


def extract_iter(query, choices, scorer=WRatio, processor=default_process, double score_cutoff=0.0, **kwargs):
    """
    Find the best match in a list of choices

    Parameters
    ----------
    query : str
        string we want to find
    choices : Iterable
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. fuzz.WRatio is used by default
    processor : Callable, optional
        Optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        Matches with a lower score than this number will be ignored. Default is 0,
        which deactivates this behaviour.
    **kwargs : Any, optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to string_metric.levenshtein

    Yields
    -------
    Tuple[str, float, Any]
        Yields similarity between the query and each choice in form of a tuple.
        The Tuple will be in the form `(<choice>, <ratio>, <index of choice>)`
        when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
        when `choices` is a mapping.
        Matches with a similarity, that is smaller than score_cutoff are skipped.
    """
    cdef int def_process = 0
    cdef scorer_context context
    cdef double score = 0.0
    cdef object py_score
    cdef size_t index

    if query is None:
        return None

    # preprocess the query
    if processor is default_process:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = processor(query)
        processor = None
    elif callable(processor):
        query = processor(query)
    elif processor:
        def_process = 1
        # since this call is only performed once it is not very expensive to
        # make it in Python
        query = default_process(query)
        processor = None
    # query might be e.g. False
    else:
        processor = None

    # directly use the C++ implementation if possible
    context = CachedScorerInit(scorer, query, def_process, kwargs)
    if context.context != NULL:
        try:
            if hasattr(choices, "items"):
                if processor is not None:
                    # c func + dict + python processor
                    for choice_key, choice in choices.items():
                        if choice is None:
                            continue

                        score = context.scorer(context.context, processor(choice), score_cutoff)

                        if score >= score_cutoff:
                            yield (choice, score, choice_key)
                else:
                    # c func + dict + no python processor
                    for choice_key, choice in choices.items():
                        if choice is None:
                            continue

                        score = context.scorer(context.context, choice, score_cutoff)

                        if score >= score_cutoff:
                            yield (choice, score, choice_key)
            else:
                index = 0
                if processor is not None:
                    # c func + list + python processor
                    for choice in choices:
                        if choice is None:
                            continue

                        score = context.scorer(context.context, processor(choice), score_cutoff)

                        if score >= score_cutoff:
                            yield (choice, score, index)
                        index += 1
                else:
                    # c func + list + no python processor
                    for choice in choices:
                        if choice is None:
                            continue

                        score = context.scorer(context.context, choice, score_cutoff)

                        if score >= score_cutoff:
                            yield (choice, score, index)
                        index += 1
        finally:
            # part of the context is dynamically allocated, so it has to be freed in any case
            context.deinit(context.context)
    # the scorer has to be called through Python
    else:
        if hasattr(choices, "items"):
            if processor is not None:
                # python func + dict + python processor
                for choice_key, choice in choices.items():
                    if choice is None:
                        continue

                    py_score = scorer(query, processor(choice),
                        processor=None, score_cutoff=score_cutoff, **kwargs)

                    if py_score >= score_cutoff:
                        yield (choice, py_score, choice_key)
            else:
                # python func + dict + no python processor
                for choice_key, choice in choices.items():
                    if choice is None:
                        continue

                    py_score = scorer(query, choice,
                        processor=None, score_cutoff=score_cutoff, **kwargs)

                    if py_score >= score_cutoff:
                        yield (choice, py_score, choice_key)
        else:
            index = 0
            if processor is not None:
                # python func + list + python processor
                for choice in choices:
                    if choice is None:
                        continue

                    py_score = scorer(query, processor(choice),
                        processor=None, score_cutoff=score_cutoff, **kwargs)

                    if py_score >= score_cutoff:
                        yield(choice, py_score, index)
                    index += 1
            else:
                # python func + list + no python processor
                for choice in choices:
                    if choice is None:
                        continue

                    py_score = scorer(query, choice,
                        processor=None, score_cutoff=score_cutoff, **kwargs)

                    if py_score >= score_cutoff:
                        yield(choice, py_score, index)
                    index += 1
