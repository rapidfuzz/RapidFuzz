import pytest

from rapidfuzz.distance import DamerauLevenshtein_cpp, DamerauLevenshtein_py
from ..common import GenericDistanceScorer

DamerauLevenshtein = GenericDistanceScorer(
    DamerauLevenshtein_py, DamerauLevenshtein_cpp
)


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
    assert (
        pytest.approx(DamerauLevenshtein.normalized_distance(left, right))
        == distance / maximum
    )

    assert (
        pytest.approx(DamerauLevenshtein.normalized_similarity(left, right))
        == similarity / maximum
    )
