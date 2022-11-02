"""
common parts of the test suite for rapidfuzz
"""
import pytest

from rapidfuzz import process_cpp, process_py


def scorer_tester(scorer, s1, s2, **kwargs):
    score1 = scorer(s1, s2, **kwargs)

    # todo currently
    if kwargs.get("score_cutoff") is None:
        score2 = process_cpp.extractOne(
            s1, [s2], processor=None, scorer=scorer, **kwargs
        )[1]
        score3 = process_cpp.extract(s1, [s2], processor=None, scorer=scorer, **kwargs)[
            0
        ][1]
        score4 = process_py.extractOne(
            s1, [s2], processor=None, scorer=scorer, **kwargs
        )[1]
        score5 = process_py.extract(s1, [s2], processor=None, scorer=scorer, **kwargs)[
            0
        ][1]
        assert pytest.approx(score1) == score2
        assert pytest.approx(score1) == score3
        assert pytest.approx(score1) == score4
        assert pytest.approx(score1) == score5

    score6 = process_cpp.cdist([s1], [s2], processor=None, scorer=scorer, **kwargs)[0][
        0
    ]
    score7 = process_py.cdist([s1], [s2], processor=None, scorer=scorer, **kwargs)[0][0]
    assert pytest.approx(score1) == score6
    assert pytest.approx(score1) == score7
    return score1


class GenericScorer:
    def __init__(self, py_scorer, cpp_scorer):
        self.py_scorer = py_scorer
        self.cpp_scorer = cpp_scorer

    def distance(self, *args, **kwargs):
        score1 = scorer_tester(self.cpp_scorer.distance, *args, **kwargs)
        score2 = scorer_tester(self.py_scorer.distance, *args, **kwargs)
        assert pytest.approx(score1) == score2
        return score1

    def similarity(self, *args, **kwargs):
        score1 = scorer_tester(self.cpp_scorer.similarity, *args, **kwargs)
        score2 = scorer_tester(self.py_scorer.similarity, *args, **kwargs)
        assert pytest.approx(score1) == score2
        return score1

    def normalized_distance(self, *args, **kwargs):
        score1 = scorer_tester(self.cpp_scorer.normalized_distance, *args, **kwargs)
        score2 = scorer_tester(self.py_scorer.normalized_distance, *args, **kwargs)
        assert pytest.approx(score1) == score2
        return score1

    def normalized_similarity(self, *args, **kwargs):
        score1 = scorer_tester(self.cpp_scorer.normalized_similarity, *args, **kwargs)
        score2 = scorer_tester(self.py_scorer.normalized_similarity, *args, **kwargs)
        assert pytest.approx(score1) == score2
        return score1
