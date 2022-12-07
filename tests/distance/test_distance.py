import pytest
from tests.distance.common import all_scorer_modules


@pytest.mark.parametrize("scorer", all_scorer_modules)
def test_none(scorer):
    """
    All normalized scorers should be able to handle None values
    """
    assert scorer.normalized_distance(None, "test") == 1.0
    assert scorer.normalized_similarity(None, "test") == 0.0


@pytest.mark.parametrize("scorer", all_scorer_modules)
def test_nan(scorer):
    """
    All normalized scorers should be able to handle float("nan")
    """
    assert scorer.normalized_distance(float("nan"), "test") == 1.0
    assert scorer.normalized_similarity(float("nan"), "test") == 0.0
