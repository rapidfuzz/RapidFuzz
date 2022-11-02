import unittest
import pytest

from rapidfuzz.distance import JaroWinkler_cpp, JaroWinkler_py
from ..common import GenericScorer, scorer_tester

JaroWinkler = GenericScorer(JaroWinkler_py, JaroWinkler_cpp)

def jarowinkler_distance(s1, s2, **kwargs):
    score1 = JaroWinkler.distance(s1, s2, **kwargs)
    score2 = JaroWinkler.normalized_distance(s1, s2, **kwargs)
    score3 = JaroWinkler.distance(s2, s1, **kwargs)
    score4 = JaroWinkler.normalized_distance(s2, s1, **kwargs)
    assert pytest.approx(score1) == score2
    assert pytest.approx(score1) == score3
    assert pytest.approx(score1) == score4
    return score1


def jarowinkler_similarity(s1, s2, **kwargs):
    score1 = JaroWinkler.similarity(s1, s2, **kwargs)
    score2 = JaroWinkler.normalized_similarity(s1, s2, **kwargs)
    score3 = JaroWinkler.similarity(s2, s1, **kwargs)
    score4 = JaroWinkler.normalized_similarity(s2, s1, **kwargs)
    assert pytest.approx(score1) == score2
    assert pytest.approx(score1) == score3
    assert pytest.approx(score1) == score4
    return score1

class JaroWinklerTest(unittest.TestCase):
    def test_hash_special_case(self):
        self.jaro_winkler_similarity([0, -1], [0, -2], 0.66666)

    def test_edge_case_lengths(self):
        self.jaro_winkler_similarity("", "", 0)
        self.jaro_winkler_similarity("0", "0", 1)
        self.jaro_winkler_similarity("00", "00", 1)
        self.jaro_winkler_similarity("0", "00", 0.85)

        self.jaro_winkler_similarity("0" * 65, "0" * 65, 1)
        self.jaro_winkler_similarity("0" * 64, "0" * 65, 0.9969)
        self.jaro_winkler_similarity("0" * 63, "0" * 65, 0.9938)

        s1 = "10000000000000000000000000000000000000000000000000000000000000020"
        s2 = "00000000000000000000000000000000000000000000000000000000000000000"
        self.jaro_winkler_similarity(s1, s2, 0.97948)

        s1 = "00000000000000100000000000000000000000010000000000000000000000000"
        s2 = "0000000000000000000000000000000000000000000000000000000000000000000000000000001"
        self.jaro_winkler_similarity(s2, s1, 0.95333)

        s1 = "00000000000000000000000000000000000000000000000000000000000000000"
        s2 = (
            "010000000000000000000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000000000000000"
        )
        self.jaro_winkler_similarity(s2, s1, 0.85234)
