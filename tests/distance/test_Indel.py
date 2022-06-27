import unittest

from rapidfuzz.distance import Indel_cpp, Indel_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Indel:
    @staticmethod
    def distance(s1, s2, *, processor=None, score_cutoff=None):
        dist1 = Indel_cpp.distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = Indel_py.distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(s1, s2, *, processor=None, score_cutoff=None):
        dist1 = Indel_cpp.similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = Indel_py.similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(s1, s2, processor=None, *, score_cutoff=None):
        dist1 = Indel_cpp.normalized_distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = Indel_py.normalized_distance(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(s1, s2, processor=None, *, score_cutoff=None):
        dist1 = Indel_cpp.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        dist2 = Indel_py.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        assert isclose(dist1, dist2)
        return dist1


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
