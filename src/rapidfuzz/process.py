# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.utils import _fallback_import

_mod = "rapidfuzz.process"
extract = _fallback_import(_mod, "extract")
extractOne = _fallback_import(_mod, "extractOne")
extract_iter = _fallback_import(_mod, "extract_iter")

try:
    cdist = _fallback_import("rapidfuzz.process_cdist", "cdist")
except ImportError:

    def cdist(*args, **kwargs):
        raise NotImplementedError("implementation requires numpy to be installed")
