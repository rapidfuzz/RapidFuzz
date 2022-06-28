import unittest

from rapidfuzz.distance import Indel_cpp, Indel_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Indel:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = Indel_cpp.distance(*args, **kwargs)
        dist2 = Indel_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = Indel_cpp.similarity(*args, **kwargs)
        dist2 = Indel_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = Indel_cpp.normalized_distance(*args, **kwargs)
        dist2 = Indel_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = Indel_cpp.normalized_similarity(*args, **kwargs)
        dist2 = Indel_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


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
    Test completly different strings
    """
    assert Indel.distance("aaaa", "bbbb") == 8
    assert Indel.similarity("aaaa", "bbbb") == 0
    assert Indel.normalized_distance("aaaa", "bbbb") == 1.0
    assert Indel.normalized_similarity("aaaa", "bbbb") == 0.0


def testIssue196():
    """
    Indel distance did not work correctly for score_cutoff=1
    """
    assert Indel.distance("South Korea", "North Korea") == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=4) == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=3) == 4
    assert Indel.distance("South Korea", "North Korea", score_cutoff=2) == 3
    assert Indel.distance("South Korea", "North Korea", score_cutoff=1) == 2
    assert Indel.distance("South Korea", "North Korea", score_cutoff=0) == 1


if __name__ == "__main__":
    unittest.main()
