# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .Indel_cpp import (
        distance,
        similarity,
        normalized_distance,
        normalized_similarity,
        editops,
        opcodes,
    )
except ImportError:
    from .Indel_py import (
        distance,
        similarity,
        normalized_distance,
        normalized_similarity,
        editops,
        opcodes,
    )
