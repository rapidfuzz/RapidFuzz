import pytest

from rapidfuzz.distance import DamerauLevenshtein_cpp, DamerauLevenshtein_py


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class DamerauLevenshtein:
    @staticmethod
    def distance(*args, **kwargs):
        dist1 = DamerauLevenshtein_cpp.distance(*args, **kwargs)
        dist2 = DamerauLevenshtein_py.distance(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def similarity(*args, **kwargs):
        dist1 = DamerauLevenshtein_cpp.similarity(*args, **kwargs)
        dist2 = DamerauLevenshtein_py.similarity(*args, **kwargs)
        assert dist1 == dist2
        return dist1

    @staticmethod
    def normalized_distance(*args, **kwargs):
        dist1 = DamerauLevenshtein_cpp.normalized_distance(*args, **kwargs)
        dist2 = DamerauLevenshtein_py.normalized_distance(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def normalized_similarity(*args, **kwargs):
        dist1 = DamerauLevenshtein_cpp.normalized_similarity(*args, **kwargs)
        dist2 = DamerauLevenshtein_py.normalized_similarity(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


@pytest.mark.parametrize(
    "left, right, distance, similarity",
    [
        ("test", "text", 1, 3),
        ("test", "tset", 1, 3),
        ("test", "qwy", 4, 0),
        ("test", "testit", 2, 4),
        ("test", "tesst", 1, 4),
        ("test", "tet", 1, 3),
        ("cat", "hat", 1, 2),
        ("Niall", "Neil", 3, 2),
        ("aluminum", "Catalan", 7, 1),
        ("ATCG", "TAGC", 2, 2),
        ("ab", "ba", 1, 1),
        ("ab", "cde", 3, 0),
        ("ab", "ac", 1, 1),
        ("ab", "ba", 1, 1),
        ("ab", "bc", 2, 0),
        ("ca", "abc", 2, 1),
    ],
)
def test_distance(left, right, distance, similarity):
    maximum = max(len(left), len(right))
    assert DamerauLevenshtein.distance(left, right) == distance
    assert DamerauLevenshtein.similarity(left, right) == similarity
    assert isclose(
        DamerauLevenshtein.normalized_distance(left, right), distance / maximum
    )
    assert isclose(
        DamerauLevenshtein.normalized_similarity(left, right), similarity / maximum
    )
