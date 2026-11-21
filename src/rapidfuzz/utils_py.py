# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

import re
from unicodedata import category

_alnum_regex = re.compile(r"(?ui)\W")


def default_process(sentence: str) -> str:
    """
    This function preprocesses a string by:

    * removing all non alphanumeric characters

    * trimming whitespaces

    * converting all characters to lower case

    Parameters
    ----------
    sentence : str
        String to preprocess

    Returns
    -------
    processed_string : str
        processed string
    """
    string_out = _alnum_regex.sub(" ", sentence)
    return string_out.strip().lower()


def default_process_with_marks(sentence: str) -> str:
    """
    Preprocess text while preserving Unicode marks, including vowel signs.

    Keep alphanumeric characters and Unicode marks (Mn, Mc, and Me).
    Keep zero-width joiners and non-joiners when each adjacent character
    is a letter or mark. Replace other characters, including underscores,
    with spaces. Strip leading and trailing whitespace and lowercase the
    result with :meth:`str.lower`. Apply Unicode normalization separately
    if needed.

    Parameters
    ----------
    sentence : str
        String to preprocess

    Returns
    -------
    processed_string : str
        Processed string
    """
    if not isinstance(sentence, str):
        msg = "sentence must be a string"
        raise TypeError(msg)

    processed = []
    for index, char in enumerate(sentence):
        keep = char.isalnum() or category(char).startswith("M")
        if not keep and char in "\u200c\u200d" and 0 < index < len(sentence) - 1:
            neighbors = (sentence[index - 1], sentence[index + 1])
            keep = all(neighbor.isalpha() or category(neighbor).startswith("M") for neighbor in neighbors)
        processed.append(char if keep else " ")

    return "".join(processed).strip().lower()
