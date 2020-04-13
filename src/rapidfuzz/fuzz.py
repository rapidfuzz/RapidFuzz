# SPDX-License-Identifier: MIT
# Copyright © 2020 Max Bachmann
# Copyright © 2011 Adam Cohen

from rapidfuzz._fuzz import *
import rapidfuzz._fuzz
from typing import Union, Callable, Tuple
from rapidfuzz import utils

def ratio(s1: str, s2: str, processor: Union[bool, Callable] = False, score_cutoff: float = 0) -> float:
    """
    calculates a simple ratio between two strings

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.ratio("this is a test", "this is a test!")
        96.55171966552734
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.ratio(s1, s2, score_cutoff=score_cutoff)


def partial_ratio(s1: str, s2: str, processor: Union[bool, Callable] = False, score_cutoff: float = 0) -> float:
    """
    calculates a partial ratio between two strings

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.partial_ratio("this is a test", "this is a test!")
        100.0
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.partial_ratio(s1, s2, score_cutoff=score_cutoff)


def token_sort_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    sorts the words in the string and calculates the fuzz.ratio between them

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
        100.0
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.token_sort_ratio(s1, s2, score_cutoff=score_cutoff)


def partial_token_sort_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.partial_token_sort_ratio(s1, s2, score_cutoff=score_cutoff)


def token_set_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    Compares the words in the strings based on unique and common words between them using fuzz.ratio

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        83.8709716796875
        >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        100.0
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.token_set_ratio(s1, s2, score_cutoff=score_cutoff)


def partial_token_set_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    Compares the words in the strings based on unique and common words between them using fuzz.partial_ratio

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.partial_token_set_ratio(s1, s1, score_cutoff=score_cutoff)


def token_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
        (faster than manually executing the two functions)

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.token_ratio(s1, s2, score_cutoff=score_cutoff)


def partial_token_ratio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and fuzz.partial_token_sort_ratio
        (faster than manually executing the two functions)

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.partial_token_ratio(s1, s2, score_cutoff=score_cutoff)


def WRatio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Args:
        s1 (str): first string to compare
        s2 (str): first string to compare
        processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    """

    if callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)
    elif processor:
        s1 = utils.default_process(s1)
        s2 = utils.default_process(s2)

    return rapidfuzz._fuzz.WRatio(s1, s2, score_cutoff=score_cutoff)


def QRatio(s1: str, s2: str, processor: Union[bool, Callable] = True, score_cutoff: float = 0) -> float:
    """
    calculates a quick ratio between two strings using fuzz.ratio

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
		processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process
            is used by default, which lowercases the strings and trims whitespace
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    """

    return ratio(s1, s2, processor, score_cutoff=score_cutoff)