"""
common parts of the test suite for rapidfuzz
"""
import pytest

from rapidfuzz import process_cpp, process_py
from rapidfuzz import utils


def scorer_tester(scorer, s1, s2, **kwargs):
    score1 = scorer(s1, s2, **kwargs)

    if s1 is None or s2 is None:
        return score1

    if "processor" not in kwargs:
        kwargs["processor"] = None
    elif kwargs["processor"] is True:
        kwargs["processor"] = utils.default_process
    elif kwargs["processor"] is False:
        kwargs["processor"] = None

    # todo currently
    if kwargs.get("score_cutoff") is None:
        score2 = process_cpp.extractOne(s1, [s2], scorer=scorer, **kwargs)[1]
        score3 = process_cpp.extract(s1, [s2], scorer=scorer, **kwargs)[0][1]
        score4 = process_py.extractOne(s1, [s2], scorer=scorer, **kwargs)[1]
        score5 = process_py.extract(s1, [s2], scorer=scorer, **kwargs)[0][1]
        assert pytest.approx(score1) == score2
        assert pytest.approx(score1) == score3
        assert pytest.approx(score1) == score4
        assert pytest.approx(score1) == score5

    score6 = process_cpp.cdist([s1], [s2], scorer=scorer, **kwargs)[0][0]
    score7 = process_py.cdist([s1], [s2], scorer=scorer, **kwargs)[0][0]
    assert pytest.approx(score1) == score6
    assert pytest.approx(score1) == score7
    return score1


class GenericScorer:
    def __init__(self, py_scorer, cpp_scorer):
        self.py_scorer = py_scorer
        self.cpp_scorer = cpp_scorer

    def register_func(self, name):
        def func(*args, **kwargs):
            score1 = scorer_tester(getattr(self.cpp_scorer, name), *args, **kwargs)
            score2 = scorer_tester(getattr(self.py_scorer, name), *args, **kwargs)
            assert pytest.approx(score1) == score2
            return score1

        setattr(self, name, func)


class GenericDistanceScorer(GenericScorer):
    def __init__(self, py_scorer, cpp_scorer):
        super().__init__(py_scorer, cpp_scorer)
        self.register_func("distance")
        self.register_func("similarity")
        self.register_func("normalized_distance")
        self.register_func("normalized_similarity")
