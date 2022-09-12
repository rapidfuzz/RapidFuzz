# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz._utils import fallback_import as _fallback_import


default_process = _fallback_import("rapidfuzz.utils", "default_process")
