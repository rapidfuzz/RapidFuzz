# SPDX-License-Identifier: MIT
# Copyright © 2020 Max Bachmann
# Copyright © 2011 Adam Cohen

from rapidfuzz import fuzz, utils
from rapidfuzz.cpp_impl import extractOne
import heapq
import numbers

def iterExtract(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, score_cutoff = 0):
    if query is None:
        return
    
    a = processor(query) if processor else query

    if hasattr(choices, "items"):
        for choice, match_choice in choices.items():
            if match_choice is None:
                continue
            b = processor(match_choice) if processor else match_choice

            score = scorer(
                a, b,
                processor=None,
                score_cutoff=score_cutoff)

            if score >= score_cutoff:
                yield (match_choice, score, choice)
    else:
        for i, choice in enumerate(choices):
            if choice is None:
                continue
            b = processor(choice) if processor else choice

            score = scorer(
                a, b,
                processor=None,
                score_cutoff=score_cutoff)

            if score >= score_cutoff:
                yield (choice, score, i)

def extract(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, limit = 5, score_cutoff = 0):
    """ 
    Find the best matches in a list of choices

    Args: 
        query (str): string we want to find
        choices (Iterable): list of all strings the query should be compared with or dict with a mapping
            {<result>: <string to compare> }
        scorer (Callable): optional callable that is used to calculate the matching score between
            the query and each choice. WRatio is used by default
        processor (Callable): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        limit (int): maximum amount of results to return
        score_cutoff (float): Optional argument for a score threshold. Matches with
            a lower score than this number will not be returned. Defaults to 0

    Returns: 
        Union[List[Tuple[str, float, Any]]]: Returns a
        list of all matches that have a `score >= score_cutoff`. The list will
        be of either `(<choice>, <ratio>, <index of choice>)` when `choices` is a list of strings
        or `(<choice>, <ratio>, <key of choice>)` when `choices` is a mapping.
    """
    results = iterExtract(query, choices, scorer, processor, score_cutoff)

    if limit is None:
        return sorted(results, key=lambda x: x[1], reverse=True)

    return heapq.nlargest(limit, results, key=lambda x: x[1])
