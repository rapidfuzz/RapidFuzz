import _rapidfuzz_cpp.process
from rapidfuzz import fuzz
from typing import Iterable, List, Tuple, Optional, Union, Callable
import heapq


def extract(query: str, choices: Iterable, scorer: Callable = None, limit: int = 5, score_cutoff: int = 0) -> List[Tuple[str, float]]:
    """ 
    Find the best matches in a list of choices

    Parameters: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    scorer (Callable): optional callable that is used to calculate the matching score between
        the query and each choice. WRatio is used by default
    limit (int): maximum amount of results to return
    score_cutoff (int): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

    Returns: 
    List[Tuple[str, float]]: returns a list of all matches that have a score > score_cutoff
  
    """
    if not scorer or scorer == fuzz.WRatio:
        return _rapidfuzz_cpp.process.extract(query, list(choices), limit, score_cutoff)

    # evaluate score inside python since scorer is a python function and so it would be required
    # to add the python layer from C++ aswell
    results = []
    for choice in choices:
        score = scorer(query, choice, score_cutoff)
        if score >= score_cutoff:
            results.append((choice, score))

    return heapq.nlargest(limit, results)


def extractOne(query: str, choices: Iterable, scorer: Callable = None, score_cutoff: int = 0) -> Optional[Tuple[str, float]]:
    """
    Find the best match in a list of choices

    Parameters: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    scorer (Callable): optional callable that is used to calculate the matching score between
        the query and each choice. WRatio is used by default
    score_cutoff (int): Optional argument for a score threshold. Matches with
            a lower score than this number will not be returned. Defaults to 0

    Returns: 
    Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there is
        no match with a score > score_cutoff
    """
    if not scorer or scorer == fuzz.WRatio:
        return _rapidfuzz_cpp.process.extractOne(query, list(choices), score_cutoff)

    # evaluate score inside python since scorer is a python function and so it would be required
    # to add the python layer from C++ aswell
    match_found = False
    result_choice = ""
    for choice in choices:
        score = scorer(query, choice, score_cutoff)
        if score >= score_cutoff:
            score_cutoff = score
            match_found = True
            result_choice = choice

    return (result_choice, score_cutoff) if match_found else None
