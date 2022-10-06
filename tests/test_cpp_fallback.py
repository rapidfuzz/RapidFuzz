# simple test to ensure the C++ version is not missing any symbols
import os

os.environ["RAPIDFUZZ_IMPLEMENTATION"] = "cpp"
import rapidfuzz  # noqa: E402, F401
