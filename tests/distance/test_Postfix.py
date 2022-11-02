from rapidfuzz.distance import Postfix_cpp, Postfix_py
from ..common import GenericDistanceScorer

Postfix = GenericDistanceScorer(Postfix_py, Postfix_cpp)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert Postfix.distance("", "") == 0
    assert Postfix.similarity("", "") == 0
    assert Postfix.normalized_distance("", "") == 0.0
    assert Postfix.normalized_similarity("", "") == 1.0


def test_similar_strings():
    """
    Test similar strings
    """
    assert Postfix.distance("test", "test") == 0
    assert Postfix.similarity("test", "test") == 4
    assert Postfix.normalized_distance("test", "test") == 0
    assert Postfix.normalized_similarity("test", "test") == 1.0


def test_different_strings():
    """
    Test completely different strings
    """
    assert Postfix.distance("aaaa", "bbbb") == 4
    assert Postfix.similarity("aaaa", "bbbb") == 0
    assert Postfix.normalized_distance("aaaa", "bbbb") == 1.0
    assert Postfix.normalized_similarity("aaaa", "bbbb") == 0.0


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
