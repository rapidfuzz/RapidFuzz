from __future__ import annotations

import pytest

from rapidfuzz.distance import (
    DamerauLevenshtein_py,
    Hamming_py,
    Indel_py,
    Jaro_py,
    JaroWinkler_py,
    LCSseq_py,
    Levenshtein_py,
    OSA_py,
    Postfix_py,
    Prefix_py,
)

int_metrics = [
    DamerauLevenshtein_py,
    Hamming_py,
    Indel_py,
    LCSseq_py,
    Levenshtein_py,
    OSA_py,
    Postfix_py,
    Prefix_py,
]

float_metrics = [Jaro_py, JaroWinkler_py]

all_metrics = int_metrics + float_metrics


@pytest.mark.parametrize("metric", int_metrics)
def test_int_distance_negative_cutoff_raises(metric):
    with pytest.raises(OverflowError):
        metric.distance("aaaa", "b", score_cutoff=-1)
    with pytest.raises(OverflowError):
        metric.similarity("aaaa", "b", score_cutoff=-1)


@pytest.mark.parametrize("metric", all_metrics)
def test_normalized_distance_out_of_range_raises(metric):
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.normalized_distance("aaaa", "b", score_cutoff=1.2)
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.normalized_distance("aaaa", "b", score_cutoff=-0.1)
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.normalized_similarity("aaaa", "b", score_cutoff=1.2)
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.normalized_similarity("aaaa", "b", score_cutoff=-0.1)


@pytest.mark.parametrize("metric", float_metrics)
def test_float_metric_out_of_range_raises(metric):
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.distance("aaaa", "b", score_cutoff=1.2)
    with pytest.raises(TypeError, match="score_cutoff has to be in the range"):
        metric.similarity("aaaa", "b", score_cutoff=1.2)


@pytest.mark.parametrize("metric", all_metrics)
def test_valid_cutoffs_unchanged(metric):
    assert metric.distance("aaaa", "b", score_cutoff=None) is not None
    metric.normalized_distance("aaaa", "b", score_cutoff=1.0)
    metric.normalized_similarity("aaaa", "b", score_cutoff=0.0)
