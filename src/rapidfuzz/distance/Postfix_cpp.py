# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.distance.metrics_cpp import postfix_distance as distance
from rapidfuzz.distance.metrics_cpp import (
    postfix_normalized_distance as normalized_distance,
)
from rapidfuzz.distance.metrics_cpp import (
    postfix_normalized_similarity as normalized_similarity,
)
from rapidfuzz.distance.metrics_cpp import postfix_similarity as similarity

__all__ = [
    "distance",
    "normalized_distance",
    "normalized_similarity",
    "similarity",
]
