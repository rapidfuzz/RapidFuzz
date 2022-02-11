import unittest

from rapidfuzz.distance import Indel

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
    Test completly different strings
    """
    assert Indel.distance("aaaa", "bbbb") == 8
    assert Indel.similarity("aaaa", "bbbb") == 0
    assert Indel.normalized_distance("aaaa", "bbbb") == 1.0
    assert Indel.normalized_similarity("aaaa", "bbbb") == 0.0

if __name__ == '__main__':
    unittest.main()

