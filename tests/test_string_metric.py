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
    assert string_metric.levenshtein("", "", weights=(1, 1, 0)) == 0
    assert string_metric.levenshtein("", "", weights=(1, 1, 2)) == 0
    assert string_metric.levenshtein("", "", weights=(1, 1, 5)) == 0
    assert string_metric.levenshtein("", "", weights=(3, 7, 5)) == 0


def test_cross_type_matching():
    """
    strings should always be interpreted in the same way
    """
    assert string_metric.levenshtein("aaaa", "aaaa") == 0
    assert string_metric.levenshtein("aaaa", ["a", "a", "a", "a"]) == 0
    # todo add support in pure python
    assert string_metric.levenshtein("aaaa", [ord("a"), ord("a"), "a", "a"]) == 0


def test_word_error_rate():
    """
    it should be possible to use levenshtein to implement a word error rate
    """
    assert string_metric.levenshtein(["aaaaa", "bbbb"], ["aaaaa", "bbbb"]) == 0
    assert string_metric.levenshtein(["aaaaa", "bbbb"], ["aaaaa", "cccc"]) == 1


def test_simple_unicode_tests():
    """
    some very simple tests using unicode with scorers
    to catch relatively obvious implementation errors
    """
    s1 = "ÁÄ"
    s2 = "ABCD"
    assert string_metric.levenshtein(s1, s2) == 4  # 2 sub + 2 ins
    assert string_metric.levenshtein(s1, s2, weights=(1, 1, 0)) == 2  # 2 sub + 2 ins
    assert (
        string_metric.levenshtein(s1, s2, weights=(1, 1, 2)) == 6
    )  # 2 del + 4 ins / 2 sub + 2 ins
    assert string_metric.levenshtein(s1, s2, weights=(1, 1, 5)) == 6  # 2 del + 4 ins
    assert string_metric.levenshtein(s1, s2, weights=(1, 7, 5)) == 12  # 2 sub + 2 ins
    assert string_metric.levenshtein(s2, s1, weights=(1, 7, 5)) == 24  # 2 sub + 2 del

    assert string_metric.levenshtein(s1, s1) == 0
    assert string_metric.levenshtein(s1, s1, weights=(1, 1, 0)) == 0
    assert string_metric.levenshtein(s1, s1, weights=(1, 1, 2)) == 0
    assert string_metric.levenshtein(s1, s1, weights=(1, 1, 5)) == 0
    assert string_metric.levenshtein(s1, s1, weights=(3, 7, 5)) == 0


def test_levenshtein_editops():
    """
    basic test for levenshtein_editops
    """
    assert string_metric.levenshtein_editops("0", "") == [("delete", 0, 0)]
    assert string_metric.levenshtein_editops("", "0") == [("insert", 0, 0)]

    assert string_metric.levenshtein_editops("00", "0") == [("delete", 1, 1)]
    assert string_metric.levenshtein_editops("0", "00") == [("insert", 1, 1)]

    assert string_metric.levenshtein_editops("qabxcd", "abycdf") == [
        ("delete", 0, 0),
        ("replace", 3, 2),
        ("insert", 6, 5),
    ]
    assert string_metric.levenshtein_editops("Lorem ipsum.", "XYZLorem ABC iPsum") == [
        ("insert", 0, 0),
        ("insert", 0, 1),
        ("insert", 0, 2),
        ("insert", 6, 9),
        ("insert", 6, 10),
        ("insert", 6, 11),
        ("insert", 6, 12),
        ("replace", 7, 14),
        ("delete", 11, 18),
    ]


def test_help():
    """
    test that all help texts can be printed without throwing an exception,
    since they are implemented in C++ aswell
    """
    help(string_metric.levenshtein)
    help(string_metric.normalized_levenshtein)
    help(string_metric.levenshtein_editops)
    help(string_metric.hamming)
    help(string_metric.normalized_hamming)
    help(string_metric.jaro_similarity)
    help(string_metric.jaro_winkler_similarity)


if __name__ == "__main__":
    unittest.main()
