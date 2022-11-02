import pytest

from rapidfuzz.distance import Jaro_cpp, Jaro_py
from ..common import GenericScorer, scorer_tester

Jaro = GenericScorer(Jaro_py, Jaro_cpp)


def jaro_distance(s1, s2, **kwargs):
    score1 = Jaro.distance(s1, s2, **kwargs)
    score2 = Jaro.normalized_distance(s1, s2, **kwargs)
    score3 = Jaro.distance(s2, s1, **kwargs)
    score4 = Jaro.normalized_distance(s2, s1, **kwargs)
    assert pytest.approx(score1) == score2
    assert pytest.approx(score1) == score3
    assert pytest.approx(score1) == score4
    return score1


def jaro_similarity(s1, s2, **kwargs):
    score1 = Jaro.similarity(s1, s2, **kwargs)
    score2 = Jaro.normalized_similarity(s1, s2, **kwargs)
    score3 = Jaro.similarity(s2, s1, **kwargs)
    score4 = Jaro.normalized_similarity(s2, s1, **kwargs)
    assert pytest.approx(score1) == score2
    assert pytest.approx(score1) == score3
    assert pytest.approx(score1) == score4
    return score1


def test_hash_special_case():
    pytest.approx(jaro_similarity([0, -1], [0, -2])) == 0.66666


def test_edge_case_lengths():
    pytest.approx(jaro_similarity("", "")) == 0
    pytest.approx(jaro_similarity("0", "0")) == 1
    pytest.approx(jaro_similarity("00", "00")) == 1
    pytest.approx(jaro_similarity("0", "00")) == 0.83333

    pytest.approx(jaro_similarity("0" * 65, "0" * 65)) == 1
    pytest.approx(jaro_similarity("0" * 64, "0" * 65)) == 0.99487
    pytest.approx(jaro_similarity("0" * 63, "0" * 65)) == 0.98974

    s1 = "10000000000000000000000000000000000000000000000000000000000000020"
    s2 = "00000000000000000000000000000000000000000000000000000000000000000"
    pytest.approx(jaro_similarity(s1, s2)) == 0.97948

    s1 = "00000000000000100000000000000000000000010000000000000000000000000"
    s2 = "0000000000000000000000000000000000000000000000000000000000000000000000000000001"
    pytest.approx(jaro_similarity(s2, s1)) == 0.92223

    s1 = "00000000000000000000000000000000000000000000000000000000000000000"
    s2 = (
        "010000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000"
    )
    pytest.approx(jaro_similarity(s2, s1)) == 0.83593
