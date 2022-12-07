from tests.distance.common import Postfix


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
