# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

from rapidfuzz._feature_detector import AVX2, SSE2, supports

if supports(AVX2):
    from rapidfuzz.fuzz_cpp_impl_avx2 import (
        ratio, partial_ratio,
        partial_ratio_alignment, token_sort_ratio, token_set_ratio,
        token_ratio, partial_token_ratio, partial_token_sort_ratio, partial_token_set_ratio,
        WRatio, QRatio
    )
else:
    from rapidfuzz.fuzz_cpp_impl import (
        ratio, partial_ratio,
        partial_ratio_alignment, token_sort_ratio, token_set_ratio,
        token_ratio, partial_token_ratio, partial_token_sort_ratio, partial_token_set_ratio,
        WRatio, QRatio
    )
