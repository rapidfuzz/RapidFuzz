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
from libcpp cimport algorithm
from libcpp.utility cimport move
from libc.stdint cimport uint8_t, int32_t
from libc.math cimport floor

from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF

from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence, default_process_func

import heapq
from array import array

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


    ctypedef struct ExtractScorerComp:
        pass

    ctypedef struct ListMatchScorerElem:
        double score
        size_t index
        PyObject* choice

    ctypedef struct DictMatchScorerElem:
        double score
        size_t index
        PyObject* choice
        PyObject* key

    ctypedef struct ExtractDistanceComp:
        pass

    ctypedef struct ListMatchDistanceElem:
        size_t distance
        size_t index
        PyObject* choice

    ctypedef struct DictMatchDistanceElem:
        size_t distance
        size_t index
        PyObject* choice
        PyObject* key


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


cdef inline extractOne_dict(CachedScorerContext context, choices, processor, double score_cutoff):
    """
    implementation of extractOne for:
      - type of choices = dict
      - scorer = normalized scorer implemented in C++
    """
    cdef double score
    # use -1 as score, so even a score of 0 in the first iteration is higher
    cdef double result_score = -1
    result_choice = None
    result_key = None

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            proc_choice = processor(choice)
            if proc_choice is None:
                continue

            score = context.ratio(conv_sequence(proc_choice), score_cutoff)
        else:
            score = context.ratio(conv_sequence(choice), score_cutoff)

        if score >= score_cutoff and score > result_score:
            result_score = score_cutoff = score
            result_choice = choice
            result_key = choice_key

            if result_score == 100:
                break

    return (result_choice, result_score, result_key) if result_choice is not None else None


cdef inline extractOne_distance_dict(CachedDistanceContext context, choices, processor, size_t max_):
    """
    implementation of extractOne for:
      - type of choices = dict
      - scorer = Distance implemented in C++
    """
    cdef size_t distance
    cdef size_t result_distance = <size_t>-1
    result_choice = None
    result_key = None

    for choice_key, choice in choices.items():
        if choice is None:
            continue

        if processor is not None:
            proc_choice = processor(choice)
            if proc_choice is None:
                continue

            distance = context.ratio(conv_sequence(proc_choice), max_)
        else:
            distance = context.ratio(conv_sequence(choice), max_)

        if distance <= max_ and distance < result_distance:
            result_distance = max_ = distance
            result_choice = choice
            result_key = choice_key

            if result_distance == 0:
                break

    return (result_choice, result_distance, result_key) if result_choice is not None else None


cdef inline extractOne_list(CachedScorerContext context, choices, processor, double score_cutoff):
    """
    implementation of extractOne for:
      - type of choices = list
      - scorer = normalized scorer implemented in C++
    """
    cdef double score = 0.0
    # use -1 as score, so even a score of 0 in the first iteration is higher
    cdef double result_score = -1
    cdef size_t i
    cdef size_t result_index = 0
    result_choice = None

    for i, choice in enumerate(choices):
        if choice is None:
            continue

        if processor is not None:
            proc_choice = processor(choice)
            if proc_choice is None:
                continue

            score = context.ratio(conv_sequence(proc_choice), score_cutoff)
        else:
            score = context.ratio(conv_sequence(choice), score_cutoff)

        if score >= score_cutoff and score > result_score:
            result_score = score_cutoff = score
            result_choice = choice
            result_index = i

            if result_score == 100:
                break

    return (result_choice, result_score, result_index) if result_choice is not None else None


cdef inline extractOne_distance_list(CachedDistanceContext context, choices, processor, size_t max_):
    """
    implementation of extractOne for:
      - type of choices = list
      - scorer = Distance implemented in C++
    """
    cdef size_t distance
    cdef size_t result_distance = <size_t>-1
    cdef size_t i
    cdef size_t result_index = 0
    result_choice = None

    for i, choice in enumerate(choices):
        if choice is None:
            continue

        if processor is not None:
            proc_choice = processor(choice)
            if proc_choice is None:
                continue

            distance = context.ratio(conv_sequence(proc_choice), max_)
        else:
            distance = context.ratio(conv_sequence(choice), max_)

        if distance <= max_ and distance < result_distance:
            result_distance = max_ = distance
            result_choice = choice
            result_index = i

            if result_distance == 0:
                break

    return (result_choice, result_distance, result_index) if result_choice is not None else None


cdef inline py_extractOne_dict(query, choices, scorer, processor, double score_cutoff, dict kwargs):
    result_score = -1
    result_choice = None
    result_key = None

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

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
    cdef size_t result_index = 0
    cdef size_t i
    result_score = -1
    result_choice = None

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

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
    This behavior can be changed by passing a custom function, or None/False to disable the behavior. Preprocessing
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

    cdef int def_process = 0
    cdef CachedScorerContext ScorerContext
    cdef CachedDistanceContext DistanceContext
    cdef double c_score_cutoff = 0.0
    cdef size_t c_max = <size_t>-1

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

    if IsIntegratedScorer(scorer):
        # directly use the C++ implementation if possible
        # normalized distance implemented in C++
        query_context = conv_sequence(query)
        ScorerContext = CachedScorerInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None:
            c_score_cutoff = score_cutoff
        if c_score_cutoff < 0 or c_score_cutoff > 100:
            raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

        if hasattr(choices, "items"):
            return extractOne_dict(move(ScorerContext), choices, processor, c_score_cutoff)
        else:
            return extractOne_list(move(ScorerContext), choices, processor, c_score_cutoff)

    if IsIntegratedDistance(scorer):
        # distance implemented in C++
        query_context = conv_sequence(query)
        DistanceContext = CachedDistanceInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None and score_cutoff != -1:
            c_max = score_cutoff

        if hasattr(choices, "items"):
            return extractOne_distance_dict(move(DistanceContext), choices, processor, c_max)
        else:
            return extractOne_distance_list(move(DistanceContext), choices, processor, c_max)

    # the scorer has to be called through Python
    if score_cutoff is not None:
        c_score_cutoff = score_cutoff

    if hasattr(choices, "items"):
        return py_extractOne_dict(query, choices, scorer, processor, c_score_cutoff, kwargs)
    else:
        return py_extractOne_list(query, choices, scorer, processor, c_score_cutoff, kwargs)


cdef inline extract_dict(CachedScorerContext context, choices, processor, size_t limit, double score_cutoff):
    cdef double score = 0.0
    cdef size_t i
    cdef vector[DictMatchScorerElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    try:
        for i, (choice_key, choice) in enumerate(choices.items()):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                score = context.ratio(conv_sequence(proc_choice), score_cutoff)
            else:
                score = context.ratio(conv_sequence(choice), score_cutoff)

            if score >= score_cutoff:
                # especially the key object might be created on the fly by e.g. pandas.Dataframe
                # so we need to ensure Python does not deallocate it
                Py_INCREF(choice)
                Py_INCREF(choice_key)
                results.push_back(DictMatchScorerElem(score, i, <PyObject*>choice, <PyObject*>choice_key))

        # due to score_cutoff not always completely filled
        if limit > results.size():
            limit = results.size()

        if limit >= results.size():
            algorithm.sort(results.begin(), results.end(), ExtractScorerComp())
        else:
            algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractScorerComp())
            results.resize(limit)

        # copy elements into Python List
        result_list = PyList_New(<Py_ssize_t>limit)
        for i in range(limit):
            result_item = (<object>results[i].choice, results[i].score, <object>results[i].key)
            Py_INCREF(result_item)
            PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    finally:
        # decref all reference counts
        for item in results:
            Py_DECREF(<object>item.choice)
            Py_DECREF(<object>item.key)

    return result_list


cdef inline extract_distance_dict(CachedDistanceContext context, choices, processor, size_t limit, size_t max_):
    cdef size_t distance
    cdef size_t i
    cdef vector[DictMatchDistanceElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    try:
        for i, (choice_key, choice) in enumerate(choices.items()):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                distance = context.ratio(conv_sequence(proc_choice), max_)
            else:
                distance = context.ratio(conv_sequence(choice), max_)

            if distance <= max_:
                # especially the key object might be created on the fly by e.g. pandas.Dataframe
                # so we need to ensure Python does not deallocate it
                Py_INCREF(choice)
                Py_INCREF(choice_key)
                results.push_back(DictMatchDistanceElem(distance, i, <PyObject*>choice, <PyObject*>choice_key))

        # due to max_ not always completely filled
        if limit > results.size():
            limit = results.size()

        if limit >= results.size():
            algorithm.sort(results.begin(), results.end(), ExtractDistanceComp())
        else:
            algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractDistanceComp())
            results.resize(limit)

        # copy elements into Python List
        result_list = PyList_New(<Py_ssize_t>limit)
        for i in range(limit):
            result_item = (<object>results[i].choice, results[i].distance, <object>results[i].key)
            Py_INCREF(result_item)
            PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    finally:
        # decref all reference counts
        for item in results:
            Py_DECREF(<object>item.choice)
            Py_DECREF(<object>item.key)

    return result_list


cdef inline extract_list(CachedScorerContext context, choices, processor, size_t limit, double score_cutoff):
    cdef double score = 0.0
    cdef size_t i
    # todo possibly a smaller vector would be good to reduce memory usage
    cdef vector[ListMatchScorerElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    try:
        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue
                
                score = context.ratio(conv_sequence(proc_choice), score_cutoff)
            else:
                score = context.ratio(conv_sequence(choice), score_cutoff)

            if score >= score_cutoff:
                Py_INCREF(choice)
                results.push_back(ListMatchScorerElem(score, i, <PyObject*>choice))

        # due to score_cutoff not always completely filled
        if limit > results.size():
            limit = results.size()

        if limit >= results.size():
            algorithm.sort(results.begin(), results.end(), ExtractScorerComp())
        else:
            algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractScorerComp())
            results.resize(limit)

        # copy elements into Python List
        result_list = PyList_New(<Py_ssize_t>limit)
        for i in range(limit):
            result_item = (<object>results[i].choice, results[i].score, results[i].index)
            Py_INCREF(result_item)
            PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    finally:
        # decref all reference counts
        for item in results:
            Py_DECREF(<object>item.choice)

    return result_list


cdef inline extract_distance_list(CachedDistanceContext context, choices, processor, size_t limit, size_t max_):
    cdef size_t distance
    cdef size_t i
    # todo possibly a smaller vector would be good to reduce memory usage
    cdef vector[ListMatchDistanceElem] results
    results.reserve(<size_t>len(choices))
    cdef list result_list

    try:
        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                distance = context.ratio(conv_sequence(proc_choice), max_)
            else:
                distance = context.ratio(conv_sequence(choice), max_)

            if distance <= max_:
                Py_INCREF(choice)
                results.push_back(ListMatchDistanceElem(distance, i, <PyObject*>choice))

        # due to max_ not always completely filled
        if limit > results.size():
            limit = results.size()

        if limit >= results.size():
            algorithm.sort(results.begin(), results.end(), ExtractDistanceComp())
        else:
            algorithm.partial_sort(results.begin(), results.begin() + <ptrdiff_t>limit, results.end(), ExtractDistanceComp())
            results.resize(limit)

        # copy elements into Python List
        result_list = PyList_New(<Py_ssize_t>limit)
        for i in range(limit):
            result_item = (<object>results[i].choice, results[i].distance, results[i].index)
            Py_INCREF(result_item)
            PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    finally:
        # decref all reference counts
        for item in results:
            Py_DECREF(<object>item.choice)

    return result_list

cdef inline py_extract_dict(query, choices, scorer, processor, size_t limit, double score_cutoff, dict kwargs):
    cdef object score = None
    # todo working directly with a list is relatively slow
    # also it is not very memory efficient to allocate space for all elements even when only
    # a part is used. This should be optimised in the future
    cdef list result_list = []

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

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


cdef inline py_extract_list(query, choices, scorer, processor, size_t limit, double score_cutoff, dict kwargs):
    cdef object score = None
    # todo working directly with a list is relatively slow
    # also it is not very memory efficient to allocate space for all elements even when only
    # a part is used. This should be optimised in the future
    cdef list result_list = []
    cdef size_t i

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

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
    cdef int def_process = 0
    cdef CachedScorerContext ScorerContext
    cdef CachedDistanceContext DistanceContext
    cdef double c_score_cutoff = 0.0
    cdef size_t c_max = <size_t>-1

    if query is None:
        return []

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

    if IsIntegratedScorer(scorer):
        # directly use the C++ implementation if possible
        query_context = conv_sequence(query)
        ScorerContext = CachedScorerInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None:
            c_score_cutoff = score_cutoff
        if c_score_cutoff < 0 or c_score_cutoff > 100:
            raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

        if hasattr(choices, "items"):
            return extract_dict(move(ScorerContext), choices, processor, limit, c_score_cutoff)
        else:
            return extract_list(move(ScorerContext), choices, processor, limit, c_score_cutoff)

    if IsIntegratedDistance(scorer):
        # distance implemented in C++
        query_context = conv_sequence(query)
        DistanceContext = CachedDistanceInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None and score_cutoff != -1:
            c_max = score_cutoff

        if hasattr(choices, "items"):
            return extract_distance_dict(move(DistanceContext), choices, processor, limit, c_max)
        else:
            return extract_distance_list(move(DistanceContext), choices, processor, limit, c_max)

    # the scorer has to be called through Python
    if score_cutoff is not None:
        c_score_cutoff = score_cutoff

    if hasattr(choices, "items"):
        return py_extract_dict(query, choices, scorer, processor, limit, c_score_cutoff, kwargs)
    else:
        return py_extract_list(query, choices, scorer, processor, limit, c_score_cutoff, kwargs)


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
    cdef int def_process = 0
    cdef CachedScorerContext ScorerContext
    cdef CachedDistanceContext DistanceContext
    cdef double c_score_cutoff = 0.0
    cdef size_t c_max = <size_t>-1

    def extract_iter_dict():
        """
        implementation of extract_iter for:
          - type of choices = dict
          - scorer = normalized scorer implemented in C++
        """
        cdef double score

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                score = ScorerContext.ratio(conv_sequence(proc_choice), c_score_cutoff)
            else:
                score = ScorerContext.ratio(conv_sequence(choice), c_score_cutoff)

            if score >= score_cutoff:
                yield (choice, score, choice_key)

    def extract_iter_list():
        """
        implementation of extract_iter for:
          - type of choices = list
          - scorer = normalized scorer implemented in C++
        """
        cdef size_t i
        cdef double score

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                score = ScorerContext.ratio(conv_sequence(proc_choice), c_score_cutoff)
            else:
                score = ScorerContext.ratio(conv_sequence(choice), c_score_cutoff)

            if score >= c_score_cutoff:
                yield (choice, score, i)

    def extract_iter_distance_dict():
        """
        implementation of extract_iter for:
          - type of choices = dict
          - scorer = distance implemented in C++
        """
        cdef size_t distance

        for choice_key, choice in choices.items():
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                distance = DistanceContext.ratio(conv_sequence(proc_choice), c_max)
            else:
                distance = DistanceContext.ratio(conv_sequence(choice), c_max)

            if distance <= c_max:
                yield (choice, distance, choice_key)

    def extract_iter_distance_list():
        """
        implementation of extract_iter for:
          - type of choices = list
          - scorer = distance implemented in C++
        """
        cdef size_t i
        cdef size_t distance

        for i, choice in enumerate(choices):
            if choice is None:
                continue

            if processor is not None:
                proc_choice = processor(choice)
                if proc_choice is None:
                    continue

                distance = DistanceContext.ratio(conv_sequence(proc_choice), c_max)
            else:
                distance = DistanceContext.ratio(conv_sequence(choice), c_max)

            if distance <= c_max:
                yield (choice, distance, i)

    def py_extract_iter_dict():
        """
        implementation of extract_iter for:
          - type of choices = dict
          - scorer = python function
        """

        kwargs["processor"] = None
        kwargs["score_cutoff"] = c_score_cutoff

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
        cdef size_t i

        kwargs["processor"] = None
        kwargs["score_cutoff"] = c_score_cutoff

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

    if IsIntegratedScorer(scorer):
        # normalized distance implemented in C++
        query_context = conv_sequence(query)
        ScorerContext = CachedScorerInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None:
            c_score_cutoff = score_cutoff
        if c_score_cutoff < 0 or c_score_cutoff > 100:
            raise TypeError("score_cutoff has to be in the range of 0.0 - 100.0")

        if hasattr(choices, "items"):
            yield from extract_iter_dict()
        else:
            yield from extract_iter_list()
        # finish generator
        return

    if IsIntegratedDistance(scorer):
        # distance implemented in C++
        query_context = conv_sequence(query)
        DistanceContext = CachedDistanceInit(scorer, query_context, def_process, kwargs)
        if score_cutoff is not None and score_cutoff != -1:
            c_max = score_cutoff

        if hasattr(choices, "items"):
            yield from extract_iter_distance_dict()
        else:
            yield from extract_iter_distance_list()
        # finish generator
        return

    # the scorer has to be called through Python
    if score_cutoff is not None:
        c_score_cutoff = score_cutoff

    if hasattr(choices, "items"):
        yield from py_extract_iter_dict()
    else:
        yield from py_extract_iter_list()
