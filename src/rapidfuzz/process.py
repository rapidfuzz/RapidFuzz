# SPDX-License-Identifier: MIT
# Copyright © 2020 Max Bachmann
# Copyright © 2011 Adam Cohen

from rapidfuzz import fuzz, utils
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
        for choice in choices:
            if choice is None:
                continue
            b = processor(choice) if processor else choice

            score = scorer(
                a, b,
                processor=None,
                score_cutoff=score_cutoff)

            if score >= score_cutoff:
                yield (choice, score)

def iterExtractIndices(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, score_cutoff = 0):
    if query is None:
        return

    a = processor(query) if processor else query

    for (i, choice) in enumerate(choices):
        if choice is None:
            continue
        b = processor(choice) if processor else choice
        score = scorer(
            a, b,
            processor=None,
            score_cutoff=score_cutoff)

        if score >= score_cutoff:
            yield (i, score)


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
        Union[List[Tuple[str, float]], List[Tuple[str, float, str]]]: Returns a
        list of all matches that have a `score >= score_cutoff`. The list will
        be of either `(, )` when `choices` is a list of strings or `(, , )` when
        `choices` is a mapping.
    """
    results = iterExtract(query, choices, scorer, processor, score_cutoff)

    if limit is None:
        return sorted(results, key=lambda x: x[1], reverse=True)

    return heapq.nlargest(limit, results, key=lambda x: x[1])


def extractIndices(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, limit = 5, score_cutoff = 0):
    """ 
    Find the best matches in a list of choices

    Args: 
        query (str): string we want to find
        choices (Iterable): list of all strings the query should be compared with
        scorer (Callable): optional callable that is used to calculate the matching score between
            the query and each choice. WRatio is used by default
        processor (Callable): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        limit (int): maximum amount of results to return
        score_cutoff (float): Optional argument for a score threshold. Matches with
            a lower score than this number will not be returned. Defaults to 0

    Returns: 
        List[Tuple[int, float]]: returns a list of all incides in the list that have a score >= score_cutoff
  
    """
    results = iterExtractIndices(query, choices, scorer, processor, score_cutoff)

    if limit is None:
        return sorted(results, key=lambda x: x[1], reverse=True)

    return heapq.nlargest(limit, results, key=lambda x: x[1])


def extractBests(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, limit = 5, score_cutoff = 0):
    return extract(query, choices, scorer, processor, limit, score_cutoff)


def extractOne(query, choices, scorer = fuzz.WRatio, processor = utils.default_process, score_cutoff = 0):
    """
    Find the best match in a list of choices

    Args: 
        query (str): string we want to find
        choices (Iterable): list of all strings the query should be compared with or dict with a mapping
            {<result>: <string to compare>}
        scorer (Callable): optional callable that is used to calculate the matching score between
            the query and each choice. WRatio is used by default
        processor (Callable): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold. Matches with
            a lower score than this number will not be returned. Defaults to 0

    Returns: 
        Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there is
            no match with a score >= score_cutoff
    """
    if query is None:
        return None

    a = processor(query) if processor else query

    result_score = None
    result_choice = ""

    if hasattr(choices, "items"):
        choice_key = ""
        for choice, match_choice in choices.items():
            if match_choice is None:
                continue
            b = processor(match_choice) if processor else match_choice

            score = scorer(
                a, b,
                processor=None,
                score_cutoff=score_cutoff)

            if score >= score_cutoff:
                # very small increment for the score_cutoff, so when multiple
                # elements have the same score the first one is used
                # only done when the score is a number
                if isinstance(score, numbers.Number):
                    score_cutoff = score + 0.00001
                    if score_cutoff > 100:
                        return (match_choice, score, choice)
                else:
                    score_cutoff = score

                result_score = score
                result_choice = match_choice
                choice_key = choice
        return (result_choice, result_score, choice_key) if not result_score is None else None
    
    for choice in choices:
        if choice is None:
            continue
        b = processor(choice) if processor else choice

        score = scorer(
            a, b,
            processor=None,
            score_cutoff=score_cutoff)

        if score >= score_cutoff:
            if isinstance(score, numbers.Number):
                score_cutoff = score + 0.00001
                if score_cutoff > 100:
                    return (choice, score)
            else:
                score_cutoff = score

            result_score = score
            result_choice = choice

    return (result_choice, result_score) if not result_score is None else None
