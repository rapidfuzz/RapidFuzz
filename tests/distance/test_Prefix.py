import unittest

from rapidfuzz.distance import Prefix_cpp, Prefix_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Prefix:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = Prefix_cpp.distance(*args, **kwargs)
        dist2 = Prefix_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = Prefix_cpp.similarity(*args, **kwargs)
        dist2 = Prefix_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = Prefix_cpp.normalized_distance(*args, **kwargs)
        dist2 = Prefix_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = Prefix_cpp.normalized_similarity(*args, **kwargs)
        dist2 = Prefix_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


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


if __name__ == "__main__":
    unittest.main()
