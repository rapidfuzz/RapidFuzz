# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann
# Copyright (C) 2011 Adam Cohen

from rapidfuzz import fuzz, utils
from rapidfuzz.cpp_impl import extractOne, extract_iter
from rapidfuzz.cpp_impl import extract as extract_cpp_impl
import heapq

def extract(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, limit = 5, score_cutoff = 0, **kwargs):
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

    Returns
    -------
    List[Tuple[str, float, Any]]
        Returns a list of all matches that have a `score >= score_cutoff`.
        The list will be of either `(<choice>, <ratio>, <index of choice>)`
        when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
        when `choices` is a mapping

    """
    try:
        return extract_cpp_impl(query, choices, scorer, processor, limit, score_cutoff)
    except TypeError:
        # custom scorers are not supported by the C++ implementation yet and probably
        # not much more efficient anyways -> fallback to python implementation
        pass

    results = extract_iter(query, choices, scorer, processor, score_cutoff)

    if limit is None:
        return sorted(results, key=lambda x: x[1], reverse=True)

    return heapq.nlargest(limit, results, key=lambda x: x[1])
