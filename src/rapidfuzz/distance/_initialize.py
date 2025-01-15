# SPDX-License-Identifier: MIT
# Copyright (C) 2025 Max Bachmann
from __future__ import annotations

import contextlib
import os

from rapidfuzz._feature_detector import AVX2, SSE2, supports

__all__ = ["Editop", "Editops", "MatchingBlock", "Opcode", "Opcodes", "ScoreAlignment"]

_impl = os.environ.get("RAPIDFUZZ_IMPLEMENTATION")
if _impl == "cpp":
    imported = False
    if supports(AVX2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance._initialize_cpp_avx2 import (  # pyright: ignore[reportMissingImports]
                Editop,
                Editops,
                MatchingBlock,
                Opcode,
                Opcodes,
                ScoreAlignment,
            )

            imported = True

    if not imported and supports(SSE2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance._initialize_cpp_sse2 import (  # pyright: ignore[reportMissingImports]
                Editop,
                Editops,
                MatchingBlock,
                Opcode,
                Opcodes,
                ScoreAlignment,
            )

            imported = True

    if not imported:
        from rapidfuzz.distance._initialize_cpp import (  # pyright: ignore[reportMissingImports]
            Editop,
            Editops,
            MatchingBlock,
            Opcode,
            Opcodes,
            ScoreAlignment,
        )
elif _impl == "python":
    from rapidfuzz.distance._initialize_py import (
        Editop,
        Editops,
        MatchingBlock,
        Opcode,
        Opcodes,
        ScoreAlignment,
    )
else:
    imported = False
    if supports(AVX2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance._initialize_cpp_avx2 import (  # pyright: ignore[reportMissingImports]
                Editop,
                Editops,
                MatchingBlock,
                Opcode,
                Opcodes,
                ScoreAlignment,
            )

            imported = True

    if not imported and supports(SSE2):
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance._initialize_cpp_sse2 import (  # pyright: ignore[reportMissingImports]
                Editop,
                Editops,
                MatchingBlock,
                Opcode,
                Opcodes,
                ScoreAlignment,
            )

            imported = True

    if not imported:
        with contextlib.suppress(ImportError):
            from rapidfuzz.distance._initialize_cpp import (  # pyright: ignore[reportMissingImports]
                Editop,
                Editops,
                MatchingBlock,
                Opcode,
                Opcodes,
                ScoreAlignment,
            )

            imported = True

    if not imported:
        from rapidfuzz.distance._initialize_py import (
            Editop,
            Editops,
            MatchingBlock,
            Opcode,
            Opcodes,
            ScoreAlignment,
        )
