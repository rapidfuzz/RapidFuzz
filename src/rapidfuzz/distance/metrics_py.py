# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

__all__ = []

from rapidfuzz.distance.OSA_py import (
    distance as osa_distance,
    normalized_distance as osa_normalized_distance,
    normalized_similarity as osa_normalized_similarity,
    similarity as osa_similarity
)

__all__ += [
    "osa_distance", "osa_normalized_distance", "osa_normalized_similarity", "osa_similarity"
]

from rapidfuzz.distance.Prefix_py import (
    distance as prefix_distance,
    normalized_distance as prefix_normalized_distance,
    normalized_similarity as prefix_normalized_similarity,
    similarity as prefix_similarity
)

__all__ += [
    "prefix_distance", "prefix_normalized_distance", "prefix_normalized_similarity", "prefix_similarity"
]

from rapidfuzz.distance.Postfix_py import (
    distance as postfix_distance,
    normalized_distance as postfix_normalized_distance,
    normalized_similarity as postfix_normalized_similarity,
    similarity as postfix_similarity
)

__all__ += [
    "postfix_distance", "postfix_normalized_distance", "postfix_normalized_similarity", "postfix_similarity"
]

from rapidfuzz.distance.Jaro_py import (
    distance as jaro_distance,
    normalized_distance as jaro_normalized_distance,
    normalized_similarity as jaro_normalized_similarity,
    similarity as jaro_similarity
)

__all__ += [
    "jaro_distance", "jaro_normalized_distance", "jaro_normalized_similarity", "jaro_similarity"
]

from rapidfuzz.distance.JaroWinkler_py import (
    distance as jaro_winkler_distance,
    normalized_distance as jaro_winkler_normalized_distance,
    normalized_similarity as jaro_winkler_normalized_similarity,
    similarity as jaro_winkler_similarity
)

__all__ += [
    "jaro_winkler_distance", "jaro_winkler_normalized_distance", "jaro_winkler_normalized_similarity", "jaro_winkler_similarity"
]

from rapidfuzz.distance.DamerauLevenshtein_py import (
    distance as damerau_levenshtein_distance,
    normalized_distance as damerau_levenshtein_normalized_distance,
    normalized_similarity as damerau_levenshtein_normalized_similarity,
    similarity as damerau_levenshtein_similarity
)

__all__ += [
    "damerau_levenshtein_distance", "damerau_levenshtein_normalized_distance",
    "damerau_levenshtein_normalized_similarity", "damerau_levenshtein_similarity"
]

from rapidfuzz.distance.Levenshtein_py import (
    distance as levenshtein_distance,
    normalized_distance as levenshtein_normalized_distance,
    normalized_similarity as levenshtein_normalized_similarity,
    similarity as levenshtein_similarity,
    editops as levenshtein_editops,
    opcodes as levenshtein_opcodes
)

__all__ += [
    "levenshtein_distance", "levenshtein_normalized_distance", "levenshtein_normalized_similarity", "levenshtein_similarity",
    "levenshtein_editops", "levenshtein_opcodes"
]

from rapidfuzz.distance.LCSseq_py import (
    distance as lcs_seq_distance,
    normalized_distance as lcs_seq_normalized_distance,
    normalized_similarity as lcs_seq_normalized_similarity,
    similarity as lcs_seq_similarity,
    editops as lcs_seq_editops,
    opcodes as lcs_seq_opcodes
)

__all__ += [
    "lcs_seq_distance", "lcs_seq_normalized_distance", "lcs_seq_normalized_similarity", "lcs_seq_similarity",
    "lcs_seq_editops", "lcs_seq_opcodes"
]

from rapidfuzz.distance.Indel_py import (
    distance as indel_distance,
    normalized_distance as indel_normalized_distance,
    normalized_similarity as indel_normalized_similarity,
    similarity as indel_similarity,
    editops as indel_editops,
    opcodes as indel_opcodes
)

__all__ += [
    "indel_distance", "indel_normalized_distance", "indel_normalized_similarity", "indel_similarity",
    "indel_editops", "indel_opcodes"
]

from rapidfuzz.distance.Hamming_py import (
    distance as hamming_distance,
    normalized_distance as hamming_normalized_distance,
    normalized_similarity as hamming_normalized_similarity,
    similarity as hamming_similarity,
    editops as hamming_editops,
    opcodes as hamming_opcodes
)

__all__ += [
    "hamming_distance", "hamming_normalized_distance", "hamming_normalized_similarity", "hamming_similarity",
    "hamming_editops", "hamming_opcodes",
]
