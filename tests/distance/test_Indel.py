from rapidfuzz.distance import Indel_cpp, Indel_py
from ..common import GenericScorer


def get_scorer_flags(s1, s2, **kwargs):
    return {"maximum": len(s1) + len(s2), "symmetric": True}


Indel = GenericScorer(Indel_py, Indel_cpp, get_scorer_flags)


def test_basic():
    assert Indel.distance("", "") == 0
    assert Indel.distance("test", "test") == 0
    assert Indel.distance("aaaa", "bbbb") == 8


def test_issue_196():
    """
    Indel distance did not work correctly for score_cutoff=1
    """
    assert Indel.distance("South Korea", "North Korea") == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=4) == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=3) == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=2) == 3
    assert Indel.distance("South Korea", "North Korea", score_cutoff=1) == 2
    assert Indel.distance("South Korea", "North Korea", score_cutoff=0) == 1
