# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

try:
    from rapidfuzz.utils_cpp import default_process
except ImportError:
    from rapidfuzz.utils_py import default_process
