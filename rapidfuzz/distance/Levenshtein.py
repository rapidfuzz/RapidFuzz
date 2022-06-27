# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .Levenshtein_cpp import *
except ImportError:
    from .Levenshtein_py import *
