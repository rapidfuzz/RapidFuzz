from __future__ import annotations

from array import array

import pytest

from rapidfuzz import fuzz, fuzz_cpp, fuzz_py, process, process_cpp, process_py
from rapidfuzz.distance import Jaro, Jaro_py, LCSseq, LCSseq_py, ScoreAlignment
from tests.distance.common import Jaro as Jaro_scorers, LCSseq as LCSseq_scorers

Jaro_cpp = Jaro_scorers.cpp_scorers[0]
LCSseq_cpp = LCSseq_scorers.cpp_scorers[0]


class TruthlessSequence(list):
    def __bool__(self):
        msg = "sequence truth must not be used to determine emptiness"
        raise ValueError(msg)


@pytest.fixture(params=["list", "tuple", "array", "truthless", "numpy", "strided"])
def sequence(request):
    if request.param == "list":
        return list
    if request.param == "tuple":
        return tuple
    if request.param == "array":
        return lambda values: array("q", values)
    if request.param == "truthless":
        return TruthlessSequence
    np = pytest.importorskip("numpy")
    if request.param == "strided":
        return lambda values: np.repeat(np.asarray(values, dtype=np.int64), 2)[::2]
    return lambda values: np.asarray(values, dtype=np.int64)


# Public route and both explicit backends must agree with the numerical oracle.
@pytest.fixture(
    params=[
        (Jaro, Jaro_py, Jaro_cpp, "similarity", 5 / 6, 0.9, 0, 1, 1, 0),
        (Jaro, Jaro_py, Jaro_cpp, "normalized_similarity", 5 / 6, 0.9, 0, 1, 1, 0),
        (Jaro, Jaro_py, Jaro_cpp, "distance", 1 / 6, 0.1, 1, 0, 0, 1),
        (Jaro, Jaro_py, Jaro_cpp, "normalized_distance", 1 / 6, 0.1, 1, 0, 0, 1),
        (LCSseq, LCSseq_py, LCSseq_cpp, "similarity", 3, 4, 0, 4, 0, TypeError),
        (LCSseq, LCSseq_py, LCSseq_cpp, "distance", 1, 0, 1, 0, 0, TypeError),
        (LCSseq, LCSseq_py, LCSseq_cpp, "normalized_similarity", 3 / 4, 0.9, 0, 1, 1, 0),
        (LCSseq, LCSseq_py, LCSseq_cpp, "normalized_distance", 1 / 4, 0.1, 1, 0, 0, 1),
        (fuzz, fuzz_py, fuzz_cpp, "partial_ratio", 600 / 7, 90, 0, 100, 100, 0),
        (fuzz, fuzz_py, fuzz_cpp, "WRatio", 75, 90, 0, 100, 0, 0),
        (fuzz, fuzz_py, fuzz_cpp, "QRatio", 75, 90, 0, 100, 0, 0),
    ],
    ids=[
        "Jaro.similarity",
        "Jaro.normalized_similarity",
        "Jaro.distance",
        "Jaro.normalized_distance",
        "LCSseq.similarity",
        "LCSseq.distance",
        "LCSseq.normalized_similarity",
        "LCSseq.normalized_distance",
        "partial_ratio",
        "WRatio",
        "QRatio",
    ],
)
def scorer_case(request):
    return request.param


@pytest.mark.parametrize("processed", [False, True])
def test_sequence_scores(sequence, scorer_case, processed):
    public, python, cpp, name, expected, cutoff, filtered, exact, empty, _ = scorer_case
    kwargs = {"processor": sequence} if processed else {}
    convert = list if processed else sequence
    a, b = convert([65, 66, 67, 68]), convert([65, 66, 67, 69])
    for module in (public, python, cpp):
        scorer = getattr(module, name)
        assert scorer(a, b, **kwargs) == pytest.approx(expected)
        assert scorer(b, a, **kwargs) == pytest.approx(expected)
        assert scorer(a, b, score_cutoff=cutoff, **kwargs) == filtered
        assert scorer(a, a, **kwargs) == exact
        assert scorer(convert([]), convert([]), **kwargs) == empty
        assert scorer(convert([]), a, **kwargs) == scorer([], [65, 66, 67, 68])
        assert scorer(a, convert([]), **kwargs) == scorer([65, 66, 67, 68], [])


def test_zero_is_not_empty(sequence, scorer_case):
    public, python, cpp, name, *_ = scorer_case
    for module in (public, python, cpp):
        scorer = getattr(module, name)
        assert scorer(sequence([0]), sequence([0])) == scorer([0], [0])


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_values(sequence, scorer_case, missing):
    public, python, cpp, name, *_, expected = scorer_case
    if expected is TypeError:
        pytest.skip("raw LCS scorers do not define missing-value scores")
    for module in (public, python, cpp):
        scorer = getattr(module, name)
        a = sequence([65, 66])
        for left, right in ((missing, a), (a, missing)):
            assert scorer(left, right) == expected


@pytest.mark.parametrize("module", [fuzz, fuzz_py, fuzz_cpp])
def test_sequence_alignment(sequence, module):
    a, b = sequence([65, 66]), sequence([70, 65, 66, 71])
    assert module.partial_ratio_alignment(a, b) == ScoreAlignment(100, 0, 2, 1, 3)
    assert module.partial_ratio_alignment(b, a) == ScoreAlignment(100, 1, 3, 0, 2)
    assert module.partial_ratio_alignment(a, b, score_cutoff=100.1) is None
    assert module.partial_ratio_alignment(sequence([]), sequence([])) == ScoreAlignment(100, 0, 0, 0, 0)
    assert module.partial_ratio_alignment(None, b) is None


@pytest.mark.parametrize("consumer", [process, process_py, process_cpp])
@pytest.mark.parametrize(
    ("scorer", "optimal"),
    [
        (Jaro.similarity, 1),
        (Jaro_py.similarity, 1),
        (LCSseq.normalized_similarity, 1),
        (LCSseq_py.normalized_similarity, 1),
        (fuzz.QRatio, 100),
        (fuzz_py.QRatio, 100),
        (fuzz_py.WRatio, 100),
        (fuzz_py.partial_ratio, 100),
    ],
)
def test_sequence_consumers(sequence, consumer, scorer, optimal):
    a, b = sequence([65, 66]), sequence([70, 71])
    match = consumer.extractOne(a, [b, a], scorer=scorer)
    assert match[0] is a
    assert match[1:] == (optimal, 1)
    assert [item[1:] for item in consumer.extract(a, [b, a], scorer=scorer)] == [(optimal, 1), (0, 0)]
    assert [item[1:] for item in consumer.extract_iter(a, [b, a], scorer=scorer)] == [(0, 0), (optimal, 1)]
    np = pytest.importorskip("numpy")
    rows = [a, b]
    np.testing.assert_allclose(consumer.cdist(rows, rows, scorer=scorer), [[optimal, 0], [0, optimal]])
    np.testing.assert_allclose(consumer.cpdist(rows, rows, scorer=scorer), [optimal, optimal])
