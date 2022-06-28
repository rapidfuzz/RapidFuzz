import unittest

from rapidfuzz.distance import LCSseq_cpp, LCSseq_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class LCSseq:
    @staticmethod
    def distance(s1, s2, *, processor=None, score_cutoff=None):
        dist1 = LCSseq_cpp.distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = LCSseq_py.distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(s1, s2, *, processor=None, score_cutoff=None):
        dist1 = LCSseq_cpp.similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = LCSseq_py.similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(s1, s2, processor=None, *, score_cutoff=None):
        dist1 = LCSseq_cpp.normalized_distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = LCSseq_py.normalized_distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(s1, s2, processor=None, *, score_cutoff=None):
        dist1 = LCSseq_cpp.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = LCSseq_py.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
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
