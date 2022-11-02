from rapidfuzz.distance import Hamming_cpp, Hamming_py
from ..common import GenericDistanceScorer

Hamming = GenericDistanceScorer(Hamming_py, Hamming_cpp)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert Hamming.distance("", "") == 0
    assert Hamming.similarity("", "") == 0
    assert Hamming.normalized_distance("", "") == 0.0
    assert Hamming.normalized_similarity("", "") == 1.0


def test_similar_strings():
    """
    Test similar strings
    """
    assert Hamming.distance("test", "test") == 0
    assert Hamming.similarity("test", "test") == 4
    assert Hamming.normalized_distance("test", "test") == 0
    assert Hamming.normalized_similarity("test", "test") == 1.0


def test_different_strings():
    """
    Test completely different strings
    """
    assert Hamming.distance("aaaa", "bbbb") == 4
    assert Hamming.similarity("aaaa", "bbbb") == 0
    assert Hamming.normalized_distance("aaaa", "bbbb") == 1.0
    assert Hamming.normalized_similarity("aaaa", "bbbb") == 0.0


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
