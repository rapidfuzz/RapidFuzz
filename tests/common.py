"""
common parts of the test suite for rapidfuzz
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any

import pytest

from rapidfuzz import process_cpp, process_py


def _get_scorer_flags_py(scorer: Any, scorer_kwargs: dict[str, Any]) -> tuple[int, int]:
    params = getattr(scorer, "_RF_ScorerPy", None)
    if params is not None:
        flags = params["get_scorer_flags"](**scorer_kwargs)
        return (flags["worst_score"], flags["optimal_score"])
    return (0, 100)


def is_none(s):
    if s is None:
        return True

    if isinstance(s, float) and isnan(s):
        return True

    return False


def scorer_tester(scorer, s1, s2, **kwargs):
    score1 = scorer(s1, s2, **kwargs)

    temp_kwargs = kwargs.copy()
    process_kwargs = {}

    if "processor" in kwargs:
        process_kwargs["processor"] = kwargs["processor"]
        del temp_kwargs["processor"]

    if "score_cutoff" in kwargs:
        process_kwargs["score_cutoff"] = kwargs["score_cutoff"]
        del temp_kwargs["score_cutoff"]

    if temp_kwargs:
        process_kwargs["scorer_kwargs"] = temp_kwargs

    extractOne_res1 = process_cpp.extractOne(s1, [s2], scorer=scorer, **process_kwargs)
    extractOne_res2 = process_py.extractOne(s1, [s2], scorer=scorer, **process_kwargs)
    extract_res1 = process_cpp.extract(s1, [s2], scorer=scorer, **process_kwargs)
    extract_res2 = process_py.extract(s1, [s2], scorer=scorer, **process_kwargs)
    extract_iter_res1 = list(process_cpp.extract_iter(s1, [s2], scorer=scorer, **process_kwargs))
    extract_iter_res2 = list(process_py.extract_iter(s1, [s2], scorer=scorer, **process_kwargs))

    if is_none(s1) or is_none(s2):
        assert extractOne_res1 is None
        assert extractOne_res2 is None
        assert extract_res1 == []
        assert extract_res2 == []
    elif kwargs.get("score_cutoff") is not None:
        worst_score, optimal_score = _get_scorer_flags_py(scorer, process_kwargs.get("scorer_kwargs", {}))
        lowest_score_worst = optimal_score > worst_score
        is_filtered = score1 < kwargs["score_cutoff"] if lowest_score_worst else score1 > kwargs["score_cutoff"]

        if is_filtered:
            assert extractOne_res1 is None
            assert extractOne_res2 is None
            assert extract_res1 == []
            assert extract_res2 == []
            assert extract_iter_res1 == []
            assert extract_iter_res2 == []
        else:
            assert pytest.approx(score1) == extractOne_res1[1]
            assert pytest.approx(score1) == extractOne_res2[1]
            assert pytest.approx(score1) == extract_res1[0][1]
            assert pytest.approx(score1) == extract_res2[0][1]
            assert pytest.approx(score1) == extract_iter_res1[0][1]
            assert pytest.approx(score1) == extract_iter_res2[0][1]
    else:
        assert pytest.approx(score1) == extractOne_res1[1]
        assert pytest.approx(score1) == extractOne_res2[1]
        assert pytest.approx(score1) == extract_res1[0][1]
        assert pytest.approx(score1) == extract_res2[0][1]
        assert pytest.approx(score1) == extract_iter_res1[0][1]
        assert pytest.approx(score1) == extract_iter_res2[0][1]

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        scores = process_cpp.cdist([s1], [s2], scorer=scorer, **process_kwargs)
        assert np.all(np.isclose(scores, score1))

        scores = process_py.cdist([s1], [s2], scorer=scorer, **process_kwargs)[0][0]
        assert np.all(np.isclose(scores, score1))

        # probably trigger multi match / simd implementations
        scores = process_cpp.cdist([s1] * 2, [s2] * 4, scorer=scorer, **process_kwargs)
        assert np.all(np.isclose(scores, score1))

        scores = process_py.cdist([s1] * 2, [s2] * 4, scorer=scorer, **process_kwargs)[0][0]
        assert np.all(np.isclose(scores, score1))

    return score1


def symmetric_scorer_tester(scorer, s1, s2, **kwargs):
    score1 = scorer_tester(scorer, s1, s2, **kwargs)
    score2 = scorer_tester(scorer, s2, s1, **kwargs)
    assert pytest.approx(score1) == score2
    return score1


@dataclass
class Scorer:
    distance: Any
    similarity: Any
    normalized_distance: Any
    normalized_similarity: Any


class GenericScorer:
    def __init__(self, py_scorers, cpp_scorers, get_scorer_flags):
        self.py_scorers = py_scorers
        self.cpp_scorers = cpp_scorers
        self.scorers = self.py_scorers + self.cpp_scorers

        def validate_attrs(func1, func2):
            assert hasattr(func1, "_RF_ScorerPy")
            assert hasattr(func2, "_RF_ScorerPy")
            assert func1.__name__ == func2.__name__
            assert func1.__qualname__ == func2.__qualname__
            assert func1.__doc__ == func2.__doc__

        for scorer in self.scorers:
            validate_attrs(scorer.distance, self.scorers[0].distance)
            validate_attrs(scorer.similarity, self.scorers[0].similarity)
            validate_attrs(scorer.normalized_distance, self.scorers[0].normalized_distance)
            validate_attrs(scorer.normalized_similarity, self.scorers[0].normalized_similarity)

        for scorer in self.cpp_scorers:
            assert hasattr(scorer.distance, "_RF_Scorer")
            assert hasattr(scorer.similarity, "_RF_Scorer")
            assert hasattr(scorer.normalized_distance, "_RF_Scorer")
            assert hasattr(scorer.normalized_similarity, "_RF_Scorer")

        self.get_scorer_flags = get_scorer_flags

    def _distance(self, s1, s2, **kwargs):
        symmetric = self.get_scorer_flags(s1, s2, **kwargs)["symmetric"]
        tester = symmetric_scorer_tester if symmetric else scorer_tester

        scores = sorted(tester(scorer.distance, s1, s2, **kwargs) for scorer in self.scorers)
        assert pytest.approx(scores[0]) == scores[-1]
        return scores[0]

    def _similarity(self, s1, s2, **kwargs):
        symmetric = self.get_scorer_flags(s1, s2, **kwargs)["symmetric"]
        tester = symmetric_scorer_tester if symmetric else scorer_tester

        scores = sorted(tester(scorer.similarity, s1, s2, **kwargs) for scorer in self.scorers)
        assert pytest.approx(scores[0]) == scores[-1]
        return scores[0]

    def _normalized_distance(self, s1, s2, **kwargs):
        symmetric = self.get_scorer_flags(s1, s2, **kwargs)["symmetric"]
        tester = symmetric_scorer_tester if symmetric else scorer_tester

        scores = sorted(tester(scorer.normalized_distance, s1, s2, **kwargs) for scorer in self.scorers)
        assert pytest.approx(scores[0]) == scores[-1]
        return scores[0]

    def _normalized_similarity(self, s1, s2, **kwargs):
        symmetric = self.get_scorer_flags(s1, s2, **kwargs)["symmetric"]
        tester = symmetric_scorer_tester if symmetric else scorer_tester

        scores = sorted(tester(scorer.normalized_similarity, s1, s2, **kwargs) for scorer in self.scorers)
        assert pytest.approx(scores[0]) == scores[-1]
        return scores[0]

    def _validate(self, s1, s2, **kwargs):
        # todo requires more complex test handling
        # score_cutoff = kwargs.get("score_cutoff")
        kwargs = {k: v for k, v in kwargs.items() if k != "score_cutoff"}

        maximum = self.get_scorer_flags(s1, s2, **kwargs)["maximum"]
        dist = self._distance(s1, s2, **kwargs)
        sim = self._similarity(s1, s2, **kwargs)
        norm_dist = self._normalized_distance(s1, s2, **kwargs)
        norm_sim = self._normalized_similarity(s1, s2, **kwargs)
        assert pytest.approx(dist) == maximum - sim
        if maximum != 0:
            assert pytest.approx(dist / maximum) == norm_dist
            assert pytest.approx(sim / maximum) == norm_sim
        else:
            assert pytest.approx(0.0) == norm_dist
            assert pytest.approx(1.0) == norm_sim

    def distance(self, s1, s2, **kwargs):
        self._validate(s1, s2, **kwargs)
        return self._distance(s1, s2, **kwargs)

    def similarity(self, s1, s2, **kwargs):
        self._validate(s1, s2, **kwargs)
        return self._similarity(s1, s2, **kwargs)

    def normalized_distance(self, s1, s2, **kwargs):
        if not is_none(s1) and not is_none(s2):
            self._validate(s1, s2, **kwargs)
        return self._normalized_distance(s1, s2, **kwargs)

    def normalized_similarity(self, s1, s2, **kwargs):
        if not is_none(s1) and not is_none(s2):
            self._validate(s1, s2, **kwargs)
        return self._normalized_similarity(s1, s2, **kwargs)
