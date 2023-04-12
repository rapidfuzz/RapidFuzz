# Pyinstaller hook to successfully freeze: https://pyinstaller.readthedocs.io/en/stable/hooks.html
from __future__ import annotations

hiddenimports = [
    "array",
    "rapidfuzz.fuzz_py",
    "rapidfuzz.fuzz_cpp",
    "rapidfuzz.utils_py",
    "rapidfuzz.utils_cpp",
    "rapidfuzz.process_py",
    "rapidfuzz.process_cpp",
    # distances
    "rapidfuzz.distance._initialize_py",
    "rapidfuzz.distance._initialize_cpp",
    "rapidfuzz.distance.metrics_cpp",
    "rapidfuzz.distance.metrics_py",
]
