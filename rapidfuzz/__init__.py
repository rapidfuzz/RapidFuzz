"""
rapid string matching library
"""
__author__ = "Max Bachmann"
__license__ = "MIT"
__version__ = "2.0.10"

from rapidfuzz import (
    process,
    distance,
    fuzz,
    string_metric,
    utils
)