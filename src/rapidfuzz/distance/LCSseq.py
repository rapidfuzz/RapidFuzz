# SPDX-License-Identifier: MIT
# Copyright (C) 2025 Max Bachmann
from __future__ import annotations

import contextlib
import os

from rapidfuzz._feature_detector import AVX2, SSE2, supports

__all__ = [
    "distance",
    "editops",
    "normalized_distance",
    "normalized_similarity",
    "opcodes",
    "similarity",
]

_impl = os.environ.get("RAPIDFUZZ_IMPLEMENTATION")
if _impl == "cpp":
    imported = False
    if supports(AVX2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance.metrics_cpp_avx2 import (  # pyright: ignore[reportMissingImports]
                lcs_seq_distance as distance,
                lcs_seq_editops as editops,
                lcs_seq_normalized_distance as normalized_distance,
                lcs_seq_normalized_similarity as normalized_similarity,
                lcs_seq_opcodes as opcodes,
                lcs_seq_similarity as similarity,
            )

            imported = True

    if not imported and supports(SSE2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance.metrics_cpp_sse2 import (  # pyright: ignore[reportMissingImports]
                lcs_seq_distance as distance,
                lcs_seq_editops as editops,
                lcs_seq_normalized_distance as normalized_distance,
                lcs_seq_normalized_similarity as normalized_similarity,
                lcs_seq_opcodes as opcodes,
                lcs_seq_similarity as similarity,
            )

            imported = True

    if not imported:
        from rapidfuzz.distance.metrics_cpp import (  # pyright: ignore[reportMissingImports]
            lcs_seq_distance as distance,
            lcs_seq_editops as editops,
            lcs_seq_normalized_distance as normalized_distance,
            lcs_seq_normalized_similarity as normalized_similarity,
            lcs_seq_opcodes as opcodes,
            lcs_seq_similarity as similarity,
        )
elif _impl == "python":
    from rapidfuzz.distance.metrics_py import (
        lcs_seq_distance as distance,
        lcs_seq_editops as editops,
        lcs_seq_normalized_distance as normalized_distance,
        lcs_seq_normalized_similarity as normalized_similarity,
        lcs_seq_opcodes as opcodes,
        lcs_seq_similarity as similarity,
    )
else:
    imported = False
    if supports(AVX2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance.metrics_cpp_avx2 import (  # pyright: ignore[reportMissingImports]
                lcs_seq_distance as distance,
                lcs_seq_editops as editops,
                lcs_seq_normalized_distance as normalized_distance,
                lcs_seq_normalized_similarity as normalized_similarity,
                lcs_seq_opcodes as opcodes,
                lcs_seq_similarity as similarity,
            )

            imported = True

    if not imported and supports(SSE2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance.metrics_cpp_sse2 import (  # pyright: ignore[reportMissingImports]
                lcs_seq_distance as distance,
                lcs_seq_editops as editops,
                lcs_seq_normalized_distance as normalized_distance,
                lcs_seq_normalized_similarity as normalized_similarity,
                lcs_seq_opcodes as opcodes,
                lcs_seq_similarity as similarity,
            )

            imported = True

    if not imported:
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance.metrics_cpp import (  # pyright: ignore[reportMissingImports]
                lcs_seq_distance as distance,
                lcs_seq_editops as editops,
                lcs_seq_normalized_distance as normalized_distance,
                lcs_seq_normalized_similarity as normalized_similarity,
                lcs_seq_opcodes as opcodes,
                lcs_seq_similarity as similarity,
            )

            imported = True

    if not imported:
        from rapidfuzz.distance.metrics_py import (
            lcs_seq_distance as distance,
            lcs_seq_editops as editops,
            lcs_seq_normalized_distance as normalized_distance,
            lcs_seq_normalized_similarity as normalized_similarity,
            lcs_seq_opcodes as opcodes,
            lcs_seq_similarity as similarity,
        )
