# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann


def _GetScorerFlagsDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 2**63 - 1, "flags": (1 << 6)}


def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 2**63 - 1, "worst_score": 0, "flags": (1 << 6)}


def _GetScorerFlagsNormalizedDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 1, "flags": (1 << 5)}


def _GetScorerFlagsNormalizedSimilarity(**kwargs):
    return {"optimal_score": 1, "worst_score": 0, "flags": (1 << 5)}


def fallback_import(module: str, name: str, set_attrs: bool = True):
    """
    import library function and possibly fall back to a pure Python version
    when no C++ implementation is available
    """
    import importlib
    import os

    impl = os.environ.get("RAPIDFUZZ_IMPLEMENTATION")

    py_mod = importlib.import_module(module + "_py")
    py_func = getattr(py_mod, name)
    if not py_func:
        raise ImportError(
            f"cannot import name '{name}' from '{py_mod.__name}' ({py_mod.__file__})"
        )

    if impl == "cpp":
        cpp_mod = importlib.import_module(module + "_cpp")
    elif impl == "python":
        return py_func
    else:
        try:
            cpp_mod = importlib.import_module(module + "_cpp")
        except ModuleNotFoundError:
            return py_func

    cpp_func = getattr(cpp_mod, name)
    if not cpp_func:
        raise ImportError(
            f"cannot import name '{name}' from '{cpp_mod.__name}' ({cpp_mod.__file__})"
        )

    # patch cpp function so help does not need to be duplicated
    if set_attrs:
        cpp_func.__name__ = py_func.__name__
        cpp_func.__doc__ = py_func.__doc__
    return cpp_func


default_distance_attribute = {"get_scorer_flags": _GetScorerFlagsDistance}
default_similarity_attribute = {"get_scorer_flags": _GetScorerFlagsSimilarity}
default_normalized_distance_attribute = {
    "get_scorer_flags": _GetScorerFlagsNormalizedDistance
}
default_normalized_similarity_attribute = {
    "get_scorer_flags": _GetScorerFlagsNormalizedSimilarity
}
