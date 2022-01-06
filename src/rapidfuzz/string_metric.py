# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

from rapidfuzz.cpp_string_metric import (
    levenshtein,
    normalized_levenshtein,
    levenshtein_editops,
    hamming,
    normalized_hamming,
    jaro_similarity,
    jaro_winkler_similarity,
    #Editops
)

import warnings

warnings.warn(
    "This module is deprecated. Use the replacements in rapidfuzz.algorithm instead",
    DeprecationWarning
)
