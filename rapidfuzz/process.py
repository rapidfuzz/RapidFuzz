# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

try:
    from rapidfuzz.process_cpp import extract, extractOne, extract_iter
except ImportError:
    from rapidfuzz.process_py import extract, extractOne, extract_iter

try:
    from rapidfuzz.process_cdist_cpp import cdist
except ImportError:
    try:
        from rapidfuzz.process_cdist_py import cdist
    except ImportError:

        def cdist(*args, **kwargs):
            raise NotImplementedError("implementation requires numpy to be installed")
