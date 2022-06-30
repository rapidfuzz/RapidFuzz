# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann


def _fallback_import(module: str, name: str):
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


default_process = _fallback_import("rapidfuzz.utils", "default_process")
