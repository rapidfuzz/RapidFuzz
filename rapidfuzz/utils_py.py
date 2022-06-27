# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

import re

_alnum_regex = re.compile(r"(?ui)\W")


def default_process(sentence):
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
    string_out = _alnum_regex.sub(" ", s)
    return string_out.strip().lower()
