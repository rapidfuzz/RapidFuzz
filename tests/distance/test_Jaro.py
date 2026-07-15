from __future__ import annotations

import pytest

from rapidfuzz import utils_cpp, utils_py
from tests.distance.common import Jaro


def test_hash_special_case():
    assert pytest.approx(Jaro.similarity([0, -1], [0, -2])) == 0.666666


def test_edge_case_lengths():
    """
    these are largely found by fuzz tests and implemented here as regression tests
    """
    assert pytest.approx(Jaro.similarity("", "")) == 1
    assert pytest.approx(Jaro.similarity("0", "0")) == 1
    assert pytest.approx(Jaro.similarity("00", "00")) == 1
    assert pytest.approx(Jaro.similarity("0", "00")) == 0.833333

    assert pytest.approx(Jaro.similarity("0" * 65, "0" * 65)) == 1
    assert pytest.approx(Jaro.similarity("0" * 64, "0" * 65)) == 0.994872
    assert pytest.approx(Jaro.similarity("0" * 63, "0" * 65)) == 0.989744

    s1 = "000000001"
    s2 = "0000010"
    assert pytest.approx(Jaro.similarity(s1, s2)) == 0.878307

    s1 = "01234567"
    s2 = "0" * 170 + "7654321" + "0" * 200
    assert pytest.approx(Jaro.similarity(s1, s2)) == 0.548740

    s1 = "10000000000000000000000000000000000000000000000000000000000000020"
    s2 = "00000000000000000000000000000000000000000000000000000000000000000"
    assert pytest.approx(Jaro.similarity(s1, s2)) == 0.979487

    s1 = "00000000000000100000000000000000000000010000000000000000000000000"
    s2 = "0000000000000000000000000000000000000000000000000000000000000000000000000000001"
    assert pytest.approx(Jaro.similarity(s2, s1)) == 0.922233

    s1 = "00000000000000000000000000000000000000000000000000000000000000000"
    s2 = (
        "010000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000"
    )
    assert pytest.approx(Jaro.similarity(s2, s1)) == 0.8359375


def test_score_cutoff():
    """
    score_cutoff was not applied to the exact similarity in the pure Python
    fallback when the coarse filters passed. The transpositions in "abcd"/"dcba"
    place the exact similarity (0.5) below the coarse upper bound (0.66)
    """
    assert pytest.approx(Jaro.similarity("abcd", "dcba")) == 0.5
    assert pytest.approx(Jaro.similarity("abcd", "dcba", score_cutoff=0.5)) == 0.5
    assert Jaro.similarity("abcd", "dcba", score_cutoff=0.6) == 0.0
    assert Jaro.normalized_similarity("abcd", "dcba", score_cutoff=0.6) == 0.0


def testCaseInsensitive():
    assert pytest.approx(Jaro.similarity("new york mets", "new YORK mets", processor=utils_cpp.default_process)) == 1.0
    assert pytest.approx(Jaro.similarity("new york mets", "new YORK mets", processor=utils_py.default_process)) == 1.0
