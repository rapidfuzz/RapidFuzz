import pytest

from rapidfuzz.distance import Jaro_cpp, Jaro_py
from ..common import GenericScorer


def get_scorer_flags(s1, s2, **kwargs):
    return {"maximum": 1.0, "symmetric": True}


Jaro = GenericScorer(Jaro_py, Jaro_cpp, get_scorer_flags)


def test_hash_special_case():
    assert pytest.approx(Jaro.similarity([0, -1], [0, -2])) == 0.666666


def test_edge_case_lengths():
    assert pytest.approx(Jaro.similarity("", "")) == 0
    assert pytest.approx(Jaro.similarity("0", "0")) == 1
    assert pytest.approx(Jaro.similarity("00", "00")) == 1
    assert pytest.approx(Jaro.similarity("0", "00")) == 0.833333

    assert pytest.approx(Jaro.similarity("0" * 65, "0" * 65)) == 1
    assert pytest.approx(Jaro.similarity("0" * 64, "0" * 65)) == 0.994872
    assert pytest.approx(Jaro.similarity("0" * 63, "0" * 65)) == 0.989744

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
