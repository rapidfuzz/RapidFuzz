# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
try:
    from .Indel_cpp import *
except ImportError:
    from .Indel_py import *
