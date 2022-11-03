from rapidfuzz.distance import Postfix_cpp, Postfix_py
from ..common import GenericScorer


def get_scorer_flags(s1, s2, **kwargs):
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


Postfix = GenericScorer(Postfix_py, Postfix_cpp, get_scorer_flags)


def test_basic():
    assert Postfix.distance("", "") == 0
    assert Postfix.distance("test", "test") == 0
    assert Postfix.distance("aaaa", "bbbb") == 4


def test_score_cutoff():
    """
    test whether score_cutoff works correctly
    """
    assert Postfix.distance("abcd", "eebcd") == 2
    assert Postfix.distance("abcd", "eebcd", score_cutoff=4) == 2
    assert Postfix.distance("abcd", "eebcd", score_cutoff=3) == 2
    assert Postfix.distance("abcd", "eebcd", score_cutoff=2) == 2
    assert Postfix.distance("abcd", "eebcd", score_cutoff=1) == 2
    assert Postfix.distance("abcd", "eebcd", score_cutoff=0) == 1
