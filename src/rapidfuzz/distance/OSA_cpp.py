# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.distance.metrics_cpp import (
    osa_distance as distance,
    osa_similarity as similarity,
    osa_normalized_distance as normalized_distance,
    osa_normalized_similarity as normalized_similarity,
)
