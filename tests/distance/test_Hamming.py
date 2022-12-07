from tests.distance.common import Hamming


def test_basic():
    assert Hamming.distance("", "") == 0
    assert Hamming.distance("test", "test") == 0
    assert Hamming.distance("aaaa", "bbbb") == 4


def test_score_cutoff():
    """
    test whether score_cutoff works correctly
    """
    assert Hamming.distance("South Korea", "North Korea") == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=4) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=3) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=2) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=1) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=0) == 1
