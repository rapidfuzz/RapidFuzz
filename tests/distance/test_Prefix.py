from tests.distance.common import Prefix


def test_basic():
    assert Prefix.distance("", "") == 0
    assert Prefix.distance("test", "test") == 0
    assert Prefix.distance("aaaa", "bbbb") == 4


def test_score_cutoff():
    """
    test whether score_cutoff works correctly
    """
    assert Prefix.distance("abcd", "abcee") == 2
    assert Prefix.distance("abcd", "abcee", score_cutoff=4) == 2
    assert Prefix.distance("abcd", "abcee", score_cutoff=3) == 2
    assert Prefix.distance("abcd", "abcee", score_cutoff=2) == 2
    assert Prefix.distance("abcd", "abcee", score_cutoff=1) == 2
    assert Prefix.distance("abcd", "abcee", score_cutoff=0) == 1
