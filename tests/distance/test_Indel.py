from rapidfuzz.distance import Indel_cpp, Indel_py
from ..common import GenericDistanceScorer

Indel = GenericDistanceScorer(Indel_py, Indel_cpp)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert Indel.distance("", "") == 0
    assert Indel.similarity("", "") == 0
    assert Indel.normalized_distance("", "") == 0.0
    assert Indel.normalized_similarity("", "") == 1.0


def test_similar_strings():
    """
    Test similar strings
    """
    assert Indel.distance("test", "test") == 0
    assert Indel.similarity("test", "test") == 8
    assert Indel.normalized_distance("test", "test") == 0
    assert Indel.normalized_similarity("test", "test") == 1.0


def test_different_strings():
    """
    Test completely different strings
    """
    assert Indel.distance("aaaa", "bbbb") == 8
    assert Indel.similarity("aaaa", "bbbb") == 0
    assert Indel.normalized_distance("aaaa", "bbbb") == 1.0
    assert Indel.normalized_similarity("aaaa", "bbbb") == 0.0


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
