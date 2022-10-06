# simple test to ensure the pure Python version is not missing any symbols
import os

os.environ["RAPIDFUZZ_IMPLEMENTATION"] = "python"
import rapidfuzz  # noqa: E402, F401
