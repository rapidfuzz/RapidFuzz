#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from rapidfuzz import process
from rapidfuzz.distance import (
    OSA_cpp,
    OSA_py,
    OSA as _OSA,
)

OSA_cpp.distance._RF_ScorerPy = _OSA.distance._RF_ScorerPy
OSA_cpp.normalized_distance._RF_ScorerPy = (
    _OSA.normalized_distance._RF_ScorerPy
)
OSA_cpp.similarity._RF_ScorerPy = _OSA.similarity._RF_ScorerPy
OSA_cpp.normalized_similarity._RF_ScorerPy = (
    _OSA.normalized_similarity._RF_ScorerPy
)
OSA_py.distance._RF_ScorerPy = _OSA.distance._RF_ScorerPy
OSA_py.normalized_distance._RF_ScorerPy = (
    _OSA.normalized_distance._RF_ScorerPy
)
OSA_py.similarity._RF_ScorerPy = _OSA.similarity._RF_ScorerPy
OSA_py.normalized_similarity._RF_ScorerPy = (
    _OSA.normalized_similarity._RF_ScorerPy
)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class CustomHashable:
    def __init__(self, string):
        self._string = string

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._string)


class OSA:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = OSA_cpp.distance(*args, **kwargs)
        dist2 = OSA_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = OSA_cpp.similarity(*args, **kwargs)
        dist2 = OSA_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = OSA_cpp.normalized_distance(*args, **kwargs)
        dist2 = OSA_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = OSA_cpp.normalized_similarity(*args, **kwargs)
        dist2 = OSA_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert OSA.distance("", "") == 0


def test_cross_type_matching():
    """
    strings should always be interpreted in the same way
    """
    assert OSA.distance("aaaa", "aaaa") == 0
    assert OSA.distance("aaaa", ["a", "a", "a", "a"]) == 0
    # todo add support in pure python
    assert OSA_cpp.distance("aaaa", [ord("a"), ord("a"), "a", "a"]) == 0
    assert OSA_cpp.distance([0, -1], [0, -2]) == 1
    assert (
        OSA_cpp.distance(
            [CustomHashable("aa"), CustomHashable("aa")],
            [CustomHashable("aa"), CustomHashable("bb")],
        )
        == 1
    )


def test_word_error_rate():
    """
    it should be possible to use levenshtein to implement a word error rate
    """
    assert OSA.distance(["aaaaa", "bbbb"], ["aaaaa", "bbbb"]) == 0
    assert OSA.distance(["aaaaa", "bbbb"], ["aaaaa", "cccc"]) == 1


def test_simple():
    """
    some simple OSA specific tests
    """
    assert OSA.distance("CA", "ABC") == 3
    assert OSA.distance("CA", "AC") == 1
    assert OSA.distance(
        "a" * 65 + "CA" + "a" * 65,
        "b" + "a" * 64 + "AC" + "a" * 64 + "b"
    ) == 3


def test_simple_unicode_tests():
    """
    some very simple tests using unicode with scorers
    to catch relatively obvious implementation errors
    """
    s1 = "ÁÄ"
    s2 = "ABCD"
    assert OSA.distance(s1, s2) == 4
    assert OSA.distance(s1, s1) == 0


if __name__ == "__main__":
    unittest.main()
