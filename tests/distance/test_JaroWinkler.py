import pytest

from rapidfuzz.distance import JaroWinkler_cpp, JaroWinkler_py
from ..common import GenericScorer


def get_scorer_flags(s1, s2, **kwargs):
    return {"maximum": 1.0, "symmetric": True}


JaroWinkler = GenericScorer(JaroWinkler_py, JaroWinkler_cpp, get_scorer_flags)


def test_hash_special_case():
    assert pytest.approx(JaroWinkler.similarity([0, -1], [0, -2])) == 0.666666


def test_edge_case_lengths():
    assert pytest.approx(JaroWinkler.similarity("", "")) == 0
    assert pytest.approx(JaroWinkler.similarity("0", "0")) == 1
    assert pytest.approx(JaroWinkler.similarity("00", "00")) == 1
    assert pytest.approx(JaroWinkler.similarity("0", "00")) == 0.85

    assert pytest.approx(JaroWinkler.similarity("0" * 65, "0" * 65)) == 1
    assert pytest.approx(JaroWinkler.similarity("0" * 64, "0" * 65)) == 0.996923
    assert pytest.approx(JaroWinkler.similarity("0" * 63, "0" * 65)) == 0.993846

    s1 = "10000000000000000000000000000000000000000000000000000000000000020"
    s2 = "00000000000000000000000000000000000000000000000000000000000000000"
    assert pytest.approx(JaroWinkler.similarity(s1, s2)) == 0.979487

    s1 = "00000000000000100000000000000000000000010000000000000000000000000"
    s2 = "0000000000000000000000000000000000000000000000000000000000000000000000000000001"
    assert pytest.approx(JaroWinkler.similarity(s2, s1)) == 0.95334

    s1 = "00000000000000000000000000000000000000000000000000000000000000000"
    s2 = (
        "010000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000"
    )
    assert pytest.approx(JaroWinkler.similarity(s2, s1)) == 0.852344
