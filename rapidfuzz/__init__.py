"""
rapid string matching library
"""
__author__ = "Max Bachmann"
__license__ = "MIT"
__version__ = "2.0.5"

from rapidfuzz import (
    process as process,
    distance as distance,
    fuzz as fuzz,
    string_metric as string_metric,
    utils as utils
)