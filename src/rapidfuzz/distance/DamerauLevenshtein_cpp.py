# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.distance.metrics_cpp import (
    damerau_levenshtein_distance as distance,
    damerau_levenshtein_similarity as similarity,
    damerau_levenshtein_normalized_distance as normalized_distance,
    damerau_levenshtein_normalized_similarity as normalized_similarity,
)
