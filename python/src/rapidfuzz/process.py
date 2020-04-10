# SPDX-License-Identifier: MIT
# Copyright © 2020 Max Bachmann
# Copyright © 2011 Adam Cohen

from rapidfuzz import fuzz, utils
from typing import Iterable, List, Tuple, Optional, Union, Callable, Generator
import heapq

def iterExtract(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            score_cutoff: float = 0) -> Generator[Tuple[str, float], None, None]:
    a = processor(query) if processor else query

    for choice in choices:
        b = processor(choice) if processor else choice

        score = scorer(
            a, b,
            processor=None,
            score_cutoff=score_cutoff)

        if score >= score_cutoff:
            yield (choice, score)


def iterExtractIndices(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            score_cutoff: float = 0) -> Generator[Tuple[int, float], None, None]:
    a = processor(query) if processor else query

    for (i, choice) in enumerate(choices):
        b = processor(choice) if processor else choice
        score = scorer(
            a, b,
            processor=None,
            score_cutoff=score_cutoff)

        if score >= score_cutoff:
            yield (i, score)


def extract(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            limit: Optional[int] = 5, score_cutoff: float = 0) -> List[Tuple[str, float]]:
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
    
    results = iterExtract(query, choices, scorer, processor, score_cutoff)

    if limit is None:
        return sorted(results, key=lambda x: x[1], reverse=True)

    return heapq.nlargest(limit, results, key=lambda x: x[1])


def extractIndices(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            limit: Optional[int] = 5, score_cutoff: float = 0) -> List[Tuple[str, float]]:
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


def extractBests(query: str, choices: Iterable, scorer: Callable = fuzz.WRatio, processor: Callable = utils.default_process,
            limit: Optional[int] = 5, score_cutoff: float = 0) -> List[Tuple[str, float]]:
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
    # evaluate score inside python since scorer is a python function and so it would be required
    # to add the python layer from C++ aswell
    a = processor(query) if processor else query

    match_found = False
    result_choice = ""

    for choice in choices:
        b = processor(choice) if processor else choice

        score = scorer(
            a, b,
            processor=None,
            score_cutoff=score_cutoff)

        if score >= score_cutoff:
            score_cutoff = score
            match_found = True
            result_choice = choice

    return (result_choice, score_cutoff) if match_found else None
