from __future__ import annotations

import pytest

from rapidfuzz import process, process_py
from rapidfuzz.distance import Levenshtein, Levenshtein_py


@pytest.mark.parametrize(
    ("s1", "s2", "weights", "dist", "sim", "norm_dist"),
    [
        ("", "", (1, 0, 1), 0, 0, 0.0),
        ("abc", "", (1, 0, 1), 0, 0, 0.0),
        ("", "abc", (1, 0, 1), 3, 0, 1.0),
        ("abc", "ac", (1, 0, 1), 0, 2, 0.0),
        ("ab", "abc", (1, 0, 1), 1, 2, 1 / 3),
        ("abc", "ax", (1, 0, 1), 1, 1, 0.5),
        ("abc", "ax", (2, 0, 3), 2, 2, 0.5),
        ("abc", "ax", (0, 0, 0), 0, 0, 0.0),
        ("abc", "ax", (1, 1, 1), 2, 1, 2 / 3),
    ],
)
@pytest.mark.parametrize("score_cutoff", [None, 0, 1])
@pytest.mark.parametrize("scorer", [Levenshtein, Levenshtein_py], ids=["public", "python"])
def test_zero_deletion_scores(s1, s2, weights, dist, sim, norm_dist, score_cutoff, scorer):
    kwargs = {"weights": weights, "score_cutoff": score_cutoff}
    expected_dist = dist if score_cutoff is None or dist <= score_cutoff else score_cutoff + 1
    expected_sim = sim if score_cutoff is None or sim >= score_cutoff else 0
    expected_norm_dist = norm_dist if score_cutoff is None or norm_dist <= score_cutoff else 1
    norm_sim = 1 - norm_dist
    expected_norm_sim = norm_sim if score_cutoff is None or norm_sim >= score_cutoff else 0
    assert scorer.distance(s1, s2, **kwargs) == expected_dist
    assert scorer.similarity(s1, s2, **kwargs) == expected_sim
    assert scorer.normalized_distance(s1, s2, **kwargs) == pytest.approx(expected_norm_dist)
    assert scorer.normalized_similarity(s1, s2, **kwargs) == pytest.approx(expected_norm_sim)


@pytest.mark.parametrize(
    ("backend", "scorer"), [(process, Levenshtein), (process_py, Levenshtein_py)], ids=["public", "python"]
)
def test_zero_deletion_process(backend, scorer):
    kwargs = {"scorer": scorer.distance, "scorer_kwargs": {"weights": (1, 0, 1)}, "score_cutoff": 0}
    expected = [("ac", 0, 1)]
    assert backend.extractOne("abc", ["ax", "ac"], **kwargs) == expected[0]
    assert backend.extract("abc", ["ax", "ac"], **kwargs) == expected
    assert list(backend.extract_iter("abc", ["ax", "ac"], **kwargs)) == expected


@pytest.mark.parametrize(
    ("backend", "scorer"), [(process, Levenshtein), (process_py, Levenshtein_py)], ids=["public", "python"]
)
def test_zero_deletion_matrices(backend, scorer):
    np = pytest.importorskip("numpy")
    kwargs = {"scorer": scorer.distance, "scorer_kwargs": {"weights": (1, 0, 1)}}
    np.testing.assert_array_equal(backend.cdist(["abc"], ["ax", "ac"], **kwargs), [[1, 0]])
    np.testing.assert_array_equal(backend.cpdist(["abc", "abc"], ["ax", "ac"], **kwargs), [1, 0])
