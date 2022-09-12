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


def fallback_import(module: str, name: str):
    import importlib
    import os

    impl = os.environ.get("RAPIDFUZZ_IMPLEMENTATION")

    if impl == "cpp":
        mod = importlib.import_module(module + "_cpp")
    elif impl == "python":
        mod = importlib.import_module(module + "_py")
    else:
        try:
            mod = importlib.import_module(module + "_cpp")
        except ModuleNotFoundError:
            mod = importlib.import_module(module + "_py")

    func = getattr(mod, name)
    if not func:
        raise ImportError(
            f"cannot import name '{name}' from '{mod.__name}' ({mod.__file__})"
        )
    return func


default_distance_attribute = {"get_scorer_flags": _GetScorerFlagsDistance}
default_similarity_attribute = {"get_scorer_flags": _GetScorerFlagsSimilarity}
default_normalized_distance_attribute = {
    "get_scorer_flags": _GetScorerFlagsNormalizedDistance
}
default_normalized_similarity_attribute = {
    "get_scorer_flags": _GetScorerFlagsNormalizedSimilarity
}
