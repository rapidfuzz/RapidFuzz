import unittest

from rapidfuzz.distance import LCSseq_cpp, LCSseq_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class LCSseq:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = LCSseq_cpp.distance(*args, **kwargs)
        dist2 = LCSseq_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = LCSseq_cpp.similarity(*args, **kwargs)
        dist2 = LCSseq_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = LCSseq_cpp.normalized_distance(*args, **kwargs)
        dist2 = LCSseq_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = LCSseq_cpp.normalized_similarity(*args, **kwargs)
        dist2 = LCSseq_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


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
    Test completly different strings
    """
    assert LCSseq.distance("aaaa", "bbbb") == 4
    assert LCSseq.similarity("aaaa", "bbbb") == 0
    assert LCSseq.normalized_distance("aaaa", "bbbb") == 1.0
    assert LCSseq.normalized_similarity("aaaa", "bbbb") == 0.0


if __name__ == "__main__":
    unittest.main()
