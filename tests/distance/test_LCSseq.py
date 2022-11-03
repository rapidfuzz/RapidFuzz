from rapidfuzz.distance import LCSseq_cpp, LCSseq_py
from ..common import GenericScorer


def get_scorer_flags(s1, s2, **kwargs):
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


LCSseq = GenericScorer(LCSseq_py, LCSseq_cpp, get_scorer_flags)


def test_basic():
    assert LCSseq.distance("", "") == 0
    assert LCSseq.distance("test", "test") == 0
    assert LCSseq.distance("aaaa", "bbbb") == 4
