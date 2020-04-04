import rapidfuzz._process
from rapidfuzz import fuzz, utils
from typing import Iterable, List, Tuple, Optional, Union, Callable
import heapq


def extract(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            limit: int = 5, score_cutoff: float = 0) -> List[Tuple[str, float]]:
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
        List[Tuple[str, float]]: returns a list of all matches that have a score >= score_cutoff
  
    """
    if (not scorer or scorer == fuzz.WRatio) and (not processor or processor == utils.default_process):
        results = rapidfuzz._process.extract(query, choices, score_cutoff, bool(processor))

    # evaluate score inside python since scorer is a python function and so it would be required
    # to add the python layer from C++ aswell
    else:
        a = processor(query) if processor else query
        results = []

        for choice in choices:
            b = processor(choice) if processor else choice

            score = scorer(a, b, score_cutoff, False)
            if score >= score_cutoff:
                results.append((choice, score))

    return heapq.nlargest(limit, results, key=lambda x: x[1])


def extractBests(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            limit: int = 5, score_cutoff: float = 0) -> List[Tuple[str, float]]:
    return extract(query, choices, scorer, processor, limit, score_cutoff)


def extractOne(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
               score_cutoff: float = 0) -> Optional[Tuple[str, float]]:
    """
    Find the best match in a list of choices

    Args: 
        query (str): string we want to find
        choices (Iterable): list of all strings the query should be compared with
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
    if (not scorer or scorer == fuzz.WRatio) and (not processor or processor == utils.default_process):
        return rapidfuzz._process.extractOne(query, choices, score_cutoff, bool(processor))

    # evaluate score inside python since scorer is a python function and so it would be required
    # to add the python layer from C++ aswell
    a = processor(query) if processor else query
    match_found = False
    result_choice = ""

    for choice in choices:
        b = processor(choice) if processor else choice

        score = scorer(a, b, score_cutoff, False)
        if score >= score_cutoff:
            score_cutoff = score
            match_found = True
            result_choice = choice

    return (result_choice, score_cutoff) if match_found else None
