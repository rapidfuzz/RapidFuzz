from rapidfuzz.distance import LCSseq_cpp, LCSseq_py
from ..common import GenericDistanceScorer

LCSseq = GenericDistanceScorer(LCSseq_py, LCSseq_cpp)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert LCSseq.distance("", "") == 0
    assert LCSseq.similarity("", "") == 0
    assert LCSseq.normalized_distance("", "") == 0
    assert LCSseq.normalized_similarity("", "") == 1.0


def test_similar_strings():
    """
    Test similar strings
    """
    assert LCSseq.distance("test", "test") == 0
    assert LCSseq.similarity("test", "test") == 4
    assert LCSseq.normalized_distance("test", "test") == 0
    assert LCSseq.normalized_similarity("test", "test") == 1.0


def test_different_strings():
    """
    Test completely different strings
    """
    assert LCSseq.distance("aaaa", "bbbb") == 4
    assert LCSseq.similarity("aaaa", "bbbb") == 0
    assert LCSseq.normalized_distance("aaaa", "bbbb") == 1.0
    assert LCSseq.normalized_similarity("aaaa", "bbbb") == 0.0
