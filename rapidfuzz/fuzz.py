# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

try:
    from rapidfuzz.fuzz_cpp import (
        ratio,
        partial_ratio,
        partial_ratio_alignment,
        token_sort_ratio,
        partial_token_sort_ratio,
        token_set_ratio,
        partial_token_set_ratio,
        token_ratio,
        partial_token_ratio,
        WRatio,
        QRatio,
    )
except ImportError:
    from rapidfuzz.fuzz_py import (
        ratio,
        partial_ratio,
        partial_ratio_alignment,
        token_sort_ratio,
        partial_token_sort_ratio,
        token_set_ratio,
        partial_token_set_ratio,
        token_ratio,
        partial_token_ratio,
        WRatio,
        QRatio,
    )
