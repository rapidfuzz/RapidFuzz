# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .LCSseq_cpp import *
except ImportError:
    from .LCSseq_py import *
