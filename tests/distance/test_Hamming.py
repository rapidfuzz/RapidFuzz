from __future__ import annotations

from rapidfuzz.distance import Hamming_py, metrics_cpp
from tests.distance.common import Hamming


def test_basic():
    assert Hamming.distance("", "") == 0
    assert Hamming.distance("test", "test") == 0
    assert Hamming.distance("aaaa", "bbbb") == 4
    assert Hamming.distance("aaaa", "aaaaa") == 1


def test_score_cutoff():
    """
    test whether score_cutoff works correctly
    """
    assert Hamming.distance("South Korea", "North Korea") == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=4) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=3) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=2) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=1) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=0) == 1


def test_Editops():
    """
    basic test for Hamming.editops
    """
    assert metrics_cpp.hamming_editops("0", "").as_list() == [("delete", 0, 0)]
    assert metrics_cpp.hamming_editops("", "0").as_list() == [("insert", 0, 0)]

    assert metrics_cpp.hamming_editops("00", "0").as_list() == [("delete", 1, 1)]
    assert metrics_cpp.hamming_editops("0", "00").as_list() == [("insert", 1, 1)]

    assert metrics_cpp.hamming_editops("qabxcd", "abycdf").as_list() == [
        ("replace", 0, 0),
        ("replace", 1, 1),
        ("replace", 2, 2),
        ("replace", 3, 3),
        ("replace", 4, 4),
        ("replace", 5, 5),
    ]

    ops = Hamming_py.editops("aaabaaa", "abbaaabba")
    assert ops.src_len == 7
    assert ops.dest_len == 9

    assert Hamming_py.editops("0", "").as_list() == [("delete", 0, 0)]
    assert Hamming_py.editops("", "0").as_list() == [("insert", 0, 0)]

    assert Hamming_py.editops("00", "0").as_list() == [("delete", 1, 1)]
    assert Hamming_py.editops("0", "00").as_list() == [("insert", 1, 1)]

    assert Hamming_py.editops("qabxcd", "abycdf").as_list() == [
        ("replace", 0, 0),
        ("replace", 1, 1),
        ("replace", 2, 2),
        ("replace", 3, 3),
        ("replace", 4, 4),
        ("replace", 5, 5),
    ]

    ops = Hamming_py.editops("aaabaaa", "abbaaabba")
    assert ops.src_len == 7
    assert ops.dest_len == 9
