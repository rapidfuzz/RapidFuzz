# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.utils import _fallback_import

_mod = "rapidfuzz.string_metric"
levenshtein = _fallback_import(_mod, "levenshtein")
normalized_levenshtein = _fallback_import(_mod, "normalized_levenshtein")
levenshtein_editops = _fallback_import(_mod, "levenshtein_editops")
hamming = _fallback_import(_mod, "hamming")
normalized_hamming = _fallback_import(_mod, "normalized_hamming")
jaro_similarity = _fallback_import(_mod, "jaro_similarity")
jaro_winkler_similarity = _fallback_import(_mod, "jaro_winkler_similarity")
