import pytest

from rapidfuzz.distance import JaroWinkler_cpp, JaroWinkler_py
from ..common import GenericDistanceScorer, scorer_tester

JaroWinkler = GenericDistanceScorer(JaroWinkler_py, JaroWinkler_cpp)


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


def test_hash_special_case():
    assert pytest.approx(jarowinkler_similarity([0, -1], [0, -2])) == 0.666666


def test_edge_case_lengths():
    assert pytest.approx(jarowinkler_similarity("", "")) == 0
    assert pytest.approx(jarowinkler_similarity("0", "0")) == 1
    assert pytest.approx(jarowinkler_similarity("00", "00")) == 1
    assert pytest.approx(jarowinkler_similarity("0", "00")) == 0.85

    assert pytest.approx(jarowinkler_similarity("0" * 65, "0" * 65)) == 1
    assert pytest.approx(jarowinkler_similarity("0" * 64, "0" * 65)) == 0.996923
    assert pytest.approx(jarowinkler_similarity("0" * 63, "0" * 65)) == 0.993846

    s1 = "10000000000000000000000000000000000000000000000000000000000000020"
    s2 = "00000000000000000000000000000000000000000000000000000000000000000"
    assert pytest.approx(jarowinkler_similarity(s1, s2)) == 0.979487

    s1 = "00000000000000100000000000000000000000010000000000000000000000000"
    s2 = "0000000000000000000000000000000000000000000000000000000000000000000000000000001"
    assert pytest.approx(jarowinkler_similarity(s2, s1)) == 0.95334

    s1 = "00000000000000000000000000000000000000000000000000000000000000000"
    s2 = (
        "010000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000"
    )
    assert pytest.approx(jarowinkler_similarity(s2, s1)) == 0.852344
