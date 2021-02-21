from itertools import product
from functools import partial
from string import ascii_letters, digits, punctuation

from hypothesis import given, assume, settings
import hypothesis.strategies as st
import pytest

from rapidfuzz import fuzz, process, utils, string_metric
import random


def levenshtein(s1, s2, weights=(1, 1, 1)):
    """
    python implementation of a generic Levenshtein distance
    this is much less error prone, than the bitparallel C implementations
    and is therefor used to test the C implementation
    """

    rows = len(s1)+1
    cols = len(s2)+1
    insert, delete, substitute = weights

    dist = [[0 for x in range(cols)] for x in range(rows)]

    for row in range(1, rows):
        dist[row][0] = row * delete

    for col in range(1, cols):
        dist[0][col] = col * insert

    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0
            else:
                cost = substitute

            dist[row][col] = min(
                dist[row-1][col] + delete,  # deletion
                dist[row][col-1] + insert,  # insertion
                dist[row-1][col-1] + cost    # substitution
            )

    return dist[-1][-1]



HYPOTHESIS_ALPHABET = ascii_letters + digits + punctuation

SCORERS = [
    fuzz.ratio,
    fuzz.partial_ratio,
    fuzz.token_set_ratio,
    fuzz.token_sort_ratio,
    fuzz.token_ratio,
    fuzz.partial_token_set_ratio,
    fuzz.partial_token_sort_ratio,
    fuzz.partial_token_ratio,
    fuzz.WRatio,
    fuzz.QRatio
]

FULL_SCORERS = [
    fuzz.ratio,
    fuzz.WRatio,
    fuzz.QRatio
]

PROCESSORS = [
    lambda x: x,
    utils.default_process
]


@given(s1=st.text(min_size=0, max_size=64), s2=st.text(min_size=0, max_size=64))
@settings(max_examples=500, deadline=None)
def test_levenshtein_word(s1, s2):
    """
    Test short Levenshtein implementation against simple implementation
    """
    assert string_metric.levenshtein(s1, s2) == levenshtein(s1, s2)
    assert string_metric.levenshtein(s1, s2, (1,1,2)) == levenshtein(s1, s2, (1,1,2))


@given(s1=st.text(min_size=65), s2=st.text(min_size=65))
@settings(max_examples=500, deadline=None)
def test_levenshtein_block(s1, s2):
    """
    Test blockwise Levenshtein implementation against simple implementation
    """
    assert string_metric.levenshtein(s1, s2) == levenshtein(s1, s2)
    assert string_metric.levenshtein(s1, s2, (1,1,2)) == levenshtein(s1, s2, (1,1,2))


@given(s1=st.text(), s2=st.text())
@settings(max_examples=500, deadline=None)
def test_levenshtein_random(s1, s2):
    """
    Test mixed strings to test through all implementations of Levenshtein
    """
    assert string_metric.levenshtein(s1, s2) == levenshtein(s1, s2)
    assert string_metric.levenshtein(s1, s2, (1,1,2)) == levenshtein(s1, s2, (1,1,2))


@given(sentence=st.text())
@settings(max_examples=200)
def test_multiple_processor_runs(sentence):
    """
    Test that running a preprocessor on a sentence
    a second time does not change the result
    """
    assert utils.default_process(sentence) \
        == utils.default_process(utils.default_process(sentence))


@pytest.mark.parametrize('scorer,processor', list(product(FULL_SCORERS, PROCESSORS)))
@given(choices=st.lists(st.text(), min_size=1))
@settings(max_examples=20, deadline=5000)
def test_only_identical_strings_extracted(scorer, processor, choices):
    """
    Test that only identical (post processing) strings score 100 on the test.
    If two strings are not identical then using full comparison methods they should
    not be a perfect (100) match.
    :param scorer:
    :param processor:
    :param data:
    :return:
    """
    query = random.choice(choices)
    assume(processor(query) != '')

    matches = process.extract(query, choices,
        scorer=scorer, processor=processor,
        score_cutoff=100, limit=None)

    assert matches != []

    for match in matches:
        assert processor(query) == processor(match[0])