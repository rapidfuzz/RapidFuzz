#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from rapidfuzz import string_metric

def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert string_metric.levenshtein("", "") == 0
    assert string_metric.levenshtein("", "", (1,1,0)) == 0
    assert string_metric.levenshtein("", "", (1,1,2)) == 0
    assert string_metric.levenshtein("", "", (1,1,5)) == 0
    assert string_metric.levenshtein("", "", (3,7,5)) == 0

def test_simple_unicode_tests():
    """
    some very simple tests using unicode with scorers
    to catch relatively obvious implementation errors
    """
    s1 = u"ÁÄ"
    s2 = "ABCD"
    assert string_metric.levenshtein(s1, s2) == 4
    assert string_metric.levenshtein(s1, s2, (1,1,0)) == 2
    assert string_metric.levenshtein(s1, s2, (1,1,2)) == 6
    assert string_metric.levenshtein(s1, s2, (1,1,5)) == 6
    assert string_metric.levenshtein(s1, s2, (3,7,5)) == 24

    assert string_metric.levenshtein(s1, s1) == 0
    assert string_metric.levenshtein(s1, s1, (1,1,0)) == 0
    assert string_metric.levenshtein(s1, s1, (1,1,2)) == 0
    assert string_metric.levenshtein(s1, s1, (1,1,5)) == 0
    assert string_metric.levenshtein(s1, s1, (3,7,5)) == 0

def test_help():
    """
    test that all help texts can be printed without throwing an exception,
    since they are implemented in C++ aswell
    """
    help(string_metric.levenshtein)
    help(string_metric.normalized_levenshtein)
    help(string_metric.hamming)
    help(string_metric.normalized_hamming)

if __name__ == '__main__':
    unittest.main()