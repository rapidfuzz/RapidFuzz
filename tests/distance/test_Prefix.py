from rapidfuzz.distance import Prefix_cpp, Prefix_py
from ..common import GenericDistanceScorer

Prefix = GenericDistanceScorer(Prefix_py, Prefix_cpp)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert Prefix.distance("", "") == 0
    assert Prefix.similarity("", "") == 0
    assert Prefix.normalized_distance("", "") == 0.0
    assert Prefix.normalized_similarity("", "") == 1.0


def test_similar_strings():
    """
    Test similar strings
    """
    assert Prefix.distance("test", "test") == 0
    assert Prefix.similarity("test", "test") == 4
    assert Prefix.normalized_distance("test", "test") == 0
    assert Prefix.normalized_similarity("test", "test") == 1.0


def test_different_strings():
    """
    Test completely different strings
    """
    assert Prefix.distance("aaaa", "bbbb") == 4
    assert Prefix.similarity("aaaa", "bbbb") == 0
    assert Prefix.normalized_distance("aaaa", "bbbb") == 1.0
    assert Prefix.normalized_similarity("aaaa", "bbbb") == 0.0


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
