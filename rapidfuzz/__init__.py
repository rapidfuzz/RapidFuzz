"""
rapid string matching library
"""
__author__: str = "Max Bachmann"
__license__: str = "MIT"
__version__: str = "2.1.0"

class RapidfuzzConfig:
    def __init__(self):
        self._allow_fallback = True

    @property
    def allow_fallback(self) -> bool:
        return self._allow_fallback

    @allow_fallback.setter
    def allow_fallback(self, value: bool):
        if value is False:
            try:
                # in rapidfuzz either all C++ versions are available or none
                # so it is enough to validate one import
                from rapidfuzz import fuzz_cpp
            except ImportError:
                raise ValueError("C++ implementation not available")
        self._allow_fallback = value

config: RapidfuzzConfig = RapidfuzzConfig()

from rapidfuzz import process, distance, fuzz, string_metric, utils
