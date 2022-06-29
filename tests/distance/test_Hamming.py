import unittest

from rapidfuzz.distance import Hamming_cpp, Hamming_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Hamming:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = Hamming_cpp.distance(*args, **kwargs)
        dist2 = Hamming_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = Hamming_cpp.similarity(*args, **kwargs)
        dist2 = Hamming_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = Hamming_cpp.normalized_distance(*args, **kwargs)
        dist2 = Hamming_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = Hamming_cpp.normalized_similarity(*args, **kwargs)
        dist2 = Hamming_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


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
    Test completly different strings
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


if __name__ == "__main__":
    unittest.main()
