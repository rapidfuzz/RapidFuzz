# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

import importlib
import os
from enum import IntFlag
from math import isnan
from typing import Any, Callable

from rapidfuzz._feature_detector import AVX2, SSE2, supports


class ScorerFlag(IntFlag):
    RESULT_F64 = 1 << 5
    RESULT_I64 = 1 << 6
    SYMMETRIC = 1 << 11


def _get_scorer_flags_distance(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 0,
        "worst_score": 2**63 - 1,
        "flags": ScorerFlag.RESULT_I64 | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_similarity(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 2**63 - 1,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_I64 | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_normalized_distance(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 0,
        "worst_score": 1,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_normalized_similarity(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 1,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


def is_none(s: Any) -> bool:
    if s is None:
        return True

    if isinstance(s, float) and isnan(s):
        return True

    return False


def add_scorer_attrs(func: Any, cached_scorer_call: dict[str, Callable[..., dict[str, Any]]]):
    func._RF_ScorerPy = cached_scorer_call
    # used to detect the function hasn't been wrapped afterwards
    func._RF_OriginalScorer = func


def optional_import_module(module: str) -> Any:
    """
    try to import module. Return None on failure
    """
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def vectorized_import(name: str) -> tuple[Any, list[Any]]:
    """
    import module best fitting for current CPU
    """
    if supports(AVX2):
        module = optional_import_module(name + "_avx2")
        if module is not None:
            return module
    if supports(SSE2):
        module = optional_import_module(name + "_sse2")
        if module is not None:
            return module

    return importlib.import_module(name)


def fallback_import(
    module: str,
    name: str,
) -> Any:
    """
    import library function and possibly fall back to a pure Python version
    when no C++ implementation is available
    """
    impl = os.environ.get("RAPIDFUZZ_IMPLEMENTATION")

    py_mod = importlib.import_module(module + "_py")
    py_func = getattr(py_mod, name)
    if not py_func:
        msg = f"cannot import name {name!r} from {py_mod.__name!r} ({py_mod.__file__})"
        raise ImportError(msg)

    if impl == "cpp":
        cpp_mod = vectorized_import(module + "_cpp")
    elif impl == "python":
        return py_func
    else:
        try:
            cpp_mod = vectorized_import(module + "_cpp")
        except Exception:
            return py_func

    cpp_func = getattr(cpp_mod, name)
    if not cpp_func:
        msg = f"cannot import name {name!r} from {cpp_mod.__name!r} ({cpp_mod.__file__})"
        raise ImportError(msg)

    return cpp_func


default_distance_attribute: dict[str, Callable[..., dict[str, Any]]] = {"get_scorer_flags": _get_scorer_flags_distance}
default_similarity_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_similarity
}
default_normalized_distance_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_normalized_distance
}
default_normalized_similarity_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_normalized_similarity
}
