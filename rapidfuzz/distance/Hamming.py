# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .Hamming_cpp import *
except ImportError:
    from .Hamming_py import *
