# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann
try:
    from rapidfuzz.string_metric_cpp import (
        levenshtein,
        normalized_levenshtein,
        levenshtein_editops,
        hamming,
        normalized_hamming,
        jaro_similarity,
        jaro_winkler_similarity,
    )
except ImportError:
    from rapidfuzz.string_metric_py import (
        levenshtein,
        normalized_levenshtein,
        levenshtein_editops,
        hamming,
        normalized_hamming,
        jaro_similarity,
        jaro_winkler_similarity,
    )
