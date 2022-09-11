#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest
from array import array

from rapidfuzz import fuzz_py, fuzz_cpp, utils
from rapidfuzz.distance import ScoreAlignment


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class fuzz:
    @staticmethod
    def ratio(*args, **kwargs):
        dist1 = fuzz_cpp.ratio(*args, **kwargs)
        dist2 = fuzz_py.ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def partial_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.partial_ratio(*args, **kwargs)
        dist2 = fuzz_py.partial_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def partial_ratio_alignment(*args, **kwargs):
        dist1 = fuzz_cpp.partial_ratio_alignment(*args, **kwargs)
        dist2 = fuzz_py.partial_ratio_alignment(*args, **kwargs)
        if dist1 is None or dist2 is None:
            assert dist1 == dist2
        else:
            assert isclose(dist1[0], dist2[0])
            assert list(dist1)[1:] == list(dist2)[1:]
        return dist1

    @staticmethod
    def token_sort_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.token_sort_ratio(*args, **kwargs)
        dist2 = fuzz_py.token_sort_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def token_set_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.token_set_ratio(*args, **kwargs)
        dist2 = fuzz_py.token_set_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def token_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.token_ratio(*args, **kwargs)
        dist2 = fuzz_py.token_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def partial_token_sort_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.partial_token_sort_ratio(*args, **kwargs)
        dist2 = fuzz_py.partial_token_sort_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def partial_token_set_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.partial_token_set_ratio(*args, **kwargs)
        dist2 = fuzz_py.partial_token_set_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def partial_token_ratio(*args, **kwargs):
        dist1 = fuzz_cpp.partial_token_ratio(*args, **kwargs)
        dist2 = fuzz_py.partial_token_ratio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def WRatio(*args, **kwargs):
        dist1 = fuzz_cpp.WRatio(*args, **kwargs)
        dist2 = fuzz_py.WRatio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1

    @staticmethod
    def QRatio(*args, **kwargs):
        dist1 = fuzz_cpp.QRatio(*args, **kwargs)
        dist2 = fuzz_py.QRatio(*args, **kwargs)
        assert isclose(dist1, dist2)
        return dist1


scorers = [
    fuzz.ratio,
    fuzz.partial_ratio,
    fuzz.token_sort_ratio,
    fuzz.token_set_ratio,
    fuzz.token_ratio,
    fuzz.partial_token_sort_ratio,
    fuzz.partial_token_set_ratio,
    fuzz.partial_token_ratio,
    fuzz.WRatio,
    fuzz.QRatio,
]

cpp_scorers = [
    fuzz_cpp.ratio,
    fuzz_cpp.partial_ratio,
    fuzz_cpp.token_sort_ratio,
    fuzz_cpp.token_set_ratio,
    fuzz_cpp.token_ratio,
    fuzz_cpp.partial_token_sort_ratio,
    fuzz_cpp.partial_token_set_ratio,
    fuzz_cpp.partial_token_ratio,
    fuzz_cpp.WRatio,
    fuzz_cpp.QRatio,
]


class RatioTest(unittest.TestCase):
    s1 = "new york mets"
    s1a = "new york mets"
    s2 = "new YORK mets"
    s3 = "the wonderful new york mets"
    s4 = "new york mets vs atlanta braves"
    s5 = "atlanta braves vs new york mets"
    s6 = "new york mets - atlanta braves"

    def testNoProcessor(self):
        self.assertEqual(fuzz.ratio(self.s1, self.s1a), 100)
        self.assertNotEqual(fuzz.ratio(self.s1, self.s2), 100)

    def testPartialRatio(self):
        self.assertEqual(fuzz.partial_ratio(self.s1, self.s3), 100)

    def testTokenSortRatio(self):
        self.assertEqual(fuzz.token_sort_ratio(self.s1, self.s1a), 100)

    def testPartialTokenSortRatio(self):
        self.assertEqual(fuzz.partial_token_sort_ratio(self.s1, self.s1a), 100)
        self.assertEqual(fuzz.partial_token_sort_ratio(self.s4, self.s5), 100)

    def testTokenSetRatio(self):
        self.assertEqual(fuzz.token_set_ratio(self.s4, self.s5), 100)

    def testPartialTokenSetRatio(self):
        self.assertEqual(fuzz.partial_token_set_ratio(self.s4, self.s5), 100)

    def testQuickRatioEqual(self):
        self.assertEqual(fuzz.QRatio(self.s1, self.s1a), 100)

    def testQuickRatioCaseInsensitive(self):
        self.assertEqual(fuzz.QRatio(self.s1, self.s2), 100)

    def testQuickRatioNotEqual(self):
        self.assertNotEqual(fuzz.QRatio(self.s1, self.s3), 100)

    def testWRatioEqual(self):
        self.assertEqual(fuzz.WRatio(self.s1, self.s1a), 100)

    def testWRatioCaseInsensitive(self):
        self.assertEqual(fuzz.WRatio(self.s1, self.s2), 100)

    def testWRatioPartialMatch(self):
        # a partial match is scaled by .9
        self.assertEqual(fuzz.WRatio(self.s1, self.s3), 90)

    def testWRatioMisorderedMatch(self):
        # misordered full matches are scaled by .95
        self.assertEqual(fuzz.WRatio(self.s4, self.s5), 95)

    def testWRatioUnicode(self):
        self.assertEqual(fuzz.WRatio(self.s1, self.s1a), 100)

    def testQRatioUnicode(self):
        self.assertEqual(fuzz.WRatio(self.s1, self.s1a), 100)

    def testIssue76(self):
        self.assertAlmostEqual(
            fuzz.partial_ratio("physics 2 vid", "study physics physics 2"),
            81.81818,
            places=4,
        )
        self.assertEqual(
            fuzz.partial_ratio("physics 2 vid", "study physics physics 2 video"), 100
        )

    def testIssue90(self):
        self.assertAlmostEqual(
            fuzz_cpp.partial_ratio("ax b", "a b a c b"), 85.71428, places=4
        )

    def testIssue138(self):
        str1 = "a" * 65
        str2 = "a" + chr(256) + "a" * 63
        self.assertAlmostEqual(fuzz.partial_ratio(str1, str2), 98.46153, places=4)

    def testPartialRatioAlignment(self):
        a = "a certain string"
        s = "certain"
        self.assertEqual(
            fuzz.partial_ratio_alignment(s, a),
            ScoreAlignment(100, 0, len(s), 2, 2 + len(s)),
        )
        self.assertEqual(
            fuzz.partial_ratio_alignment(a, s),
            ScoreAlignment(100, 2, 2 + len(s), 0, len(s)),
        )
        self.assertEqual(fuzz.partial_ratio_alignment(None, "test"), None)
        self.assertEqual(fuzz.partial_ratio_alignment("test", None), None)

        self.assertEqual(
            fuzz.partial_ratio_alignment("test", "tesx", score_cutoff=90), None
        )

    def testIssue196(self):
        """
        fuzz.WRatio did not work correctly with score_cutoffs
        """
        self.assertAlmostEqual(
            fuzz.WRatio("South Korea", "North Korea"), 81.81818, places=4
        )
        assert fuzz.WRatio("South Korea", "North Korea", score_cutoff=85.4) == 0.0
        assert fuzz.WRatio("South Korea", "North Korea", score_cutoff=85.5) == 0.0

    def testIssue231(self):
        str1 = "er merkantilismus förderte handel und verkehr mit teils marktkonformen, teils dirigistischen maßnahmen."
        str2 = "ils marktkonformen, teils dirigistischen maßnahmen. an der schwelle zum 19. jahrhundert entstand ein neu"

        alignment = fuzz.partial_ratio_alignment(str1, str2)
        self.assertEqual(alignment.src_start, 0)
        self.assertEqual(alignment.src_end, 103)
        self.assertEqual(alignment.dest_start, 0)
        self.assertEqual(alignment.dest_end, 51)


def test_empty_string():
    """
    when both strings are empty this is either a perfect match or no match
    See https://github.com/maxbachmann/RapidFuzz/issues/110
    """
    # perfect match
    assert fuzz.ratio("", "") == 100
    assert fuzz.partial_ratio("", "") == 100
    assert fuzz.token_sort_ratio("", "") == 100
    assert fuzz.partial_token_sort_ratio("", "") == 100
    assert fuzz.token_ratio("", "") == 100
    assert fuzz.partial_token_ratio("", "") == 100

    # no match
    assert fuzz.WRatio("", "") == 0
    assert fuzz.QRatio("", "") == 0
    assert fuzz.token_set_ratio("", "") == 0
    assert fuzz.partial_token_set_ratio("", "") == 0

    # perfect match when no words
    assert fuzz.token_set_ratio("    ", "    ") == 0
    assert fuzz.partial_token_set_ratio("    ", "    ") == 0


@pytest.mark.parametrize("scorer", scorers)
def test_invalid_input(scorer):
    """
    when invalid types are passed to a scorer an exception should be thrown
    """
    with pytest.raises(TypeError):
        scorer(1, 1)


@pytest.mark.parametrize("scorer", cpp_scorers)
def test_array(scorer):
    """
    arrays should be supported and treated in a compatible way to strings
    """
    # todo add support in pure python implementation
    assert scorer(array("u", RatioTest.s3), array("u", RatioTest.s3))
    assert scorer(RatioTest.s3, array("u", RatioTest.s3))
    assert scorer(array("u", RatioTest.s3), RatioTest.s3)


@pytest.mark.parametrize("scorer", scorers)
def test_none_string(scorer):
    """
    when None is passed to a scorer the result should always be 0
    """
    assert scorer("test", None) == 0
    assert scorer(None, "test") == 0


@pytest.mark.parametrize("scorer", scorers)
def test_simple_unicode_tests(scorer):
    """
    some very simple tests using unicode with scorers
    to catch relatively obvious implementation errors
    """
    s1 = "ÁÄ"
    s2 = "ABCD"
    assert scorer(s1, s2) == 0
    assert scorer(s1, s1) == 100


@pytest.mark.parametrize(
    "processor", [True, utils.default_process, lambda s: utils.default_process(s)]
)
@pytest.mark.parametrize("scorer", scorers)
def test_scorer_case_insensitive(processor, scorer):
    """
    each scorer should be able to preprocess strings properly
    """
    assert scorer(RatioTest.s1, RatioTest.s2, processor=processor) == 100


@pytest.mark.parametrize("processor", [False, None, lambda s: s])
def test_ratio_case_censitive(processor):
    assert fuzz.ratio(RatioTest.s1, RatioTest.s2, processor=processor) != 100


@pytest.mark.parametrize("scorer", scorers)
def test_custom_processor(scorer):
    """
    Any scorer should accept any type as s1 and s2, as long as it is a string
    after preprocessing.
    """
    s1 = ["chicago cubs vs new york mets", "CitiField", "2011-05-11", "8pm"]
    s2 = ["chicago cubs vs new york mets", "CitiFields", "2012-05-11", "9pm"]
    s3 = ["different string", "CitiFields", "2012-05-11", "9pm"]
    assert scorer(s1, s2, processor=lambda event: event[0]) == 100
    assert scorer(s2, s3, processor=lambda event: event[0]) != 100


@pytest.mark.parametrize("scorer", scorers)
def testIssue206(scorer):
    """
    test correct behavior of score_cutoff
    """
    score1 = scorer("South Korea", "North Korea")
    score2 = scorer("South Korea", "North Korea", score_cutoff=score1 - 0.0001)
    assert score1 == score2


@pytest.mark.parametrize("scorer", scorers)
def test_help(scorer):
    """
    test that all help texts can be printed without throwing an exception,
    since they are implemented in C++ aswell
    """
    help(scorer)


def testIssue257():
    s1 = "aaaaaaaaaaaaaaaaaaaaaaaabacaaaaaaaabaaabaaaaaaaababbbbbbbbbbabbcb"
    s2 = "aaaaaaaaaaaaaaaaaaaaaaaababaaaaaaaabaaabaaaaaaaababbbbbbbbbbabbcb"
    score = fuzz.partial_ratio(s1, s2)
    assert isclose(score, 98.46153846153847)
    score = fuzz.partial_ratio(s2, s1)
    assert isclose(score, 98.46153846153847)


if __name__ == "__main__":
    unittest.main()
