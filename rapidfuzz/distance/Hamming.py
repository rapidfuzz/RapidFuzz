# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .Hamming_cpp import (
        distance,
        similarity,
        normalized_distance,
        normalized_similarity,
    )
except ImportError:
    from .Hamming_py import (
        distance,
        similarity,
        normalized_distance,
        normalized_similarity,
    )
