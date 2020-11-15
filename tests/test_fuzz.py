#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from rapidfuzz import process, fuzz, utils

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
    fuzz.QRatio
]

class RatioTest(unittest.TestCase):
    def setUp(self):
        self.s1 = "new york mets"
        self.s1a = "new york mets"
        self.s2 = "new YORK mets"
        self.s3 = "the wonderful new york mets"
        self.s4 = "new york mets vs atlanta braves"
        self.s5 = "atlanta braves vs new york mets"
        self.s6 = "new york mets - atlanta braves"

    def testEqual(self):
        self.assertEqual(fuzz.ratio(self.s1, self.s1a),100)

    def testCaseInsensitive(self):
        self.assertNotEqual(fuzz.ratio(self.s1, self.s2),100)
        self.assertEqual(fuzz.ratio(self.s1, self.s2, processor=True),100)

    def testPartialRatio(self):
        self.assertEqual(fuzz.partial_ratio(self.s1, self.s3),100)

    def testTokenSortRatio(self):
        self.assertEqual(fuzz.token_sort_ratio(self.s1, self.s1a),100)

    def testPartialTokenSortRatio(self):
        self.assertEqual(fuzz.partial_token_sort_ratio(self.s1, self.s1a),100)
        self.assertEqual(fuzz.partial_token_sort_ratio(self.s4, self.s5),100)

    def testTokenSetRatio(self):
        self.assertEqual(fuzz.token_set_ratio(self.s4, self.s5),100)

    def testPartialTokenSetRatio(self):
        self.assertEqual(fuzz.partial_token_set_ratio(self.s4, self.s5),100)

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

    def testEmptyStrings(self):
        self.assertEqual(fuzz.ratio("", ""), 100)
        self.assertEqual(fuzz.partial_ratio("", ""), 100)

    def testNoneString(self):
        self.assertEqual(fuzz.ratio("", None), 0)
        self.assertEqual(fuzz.partial_ratio("", None), 0)

    def testWRatioUnicodeString(self):
        s1 = u"Á"
        s2 = "ABCD"
        score = fuzz.WRatio(s1, s2)
        self.assertEqual(0, score)

    def testQRatioUnicodeString(self):
        s1 = u"Á"
        s2 = "ABCD"
        score = fuzz.QRatio(s1, s2)
        self.assertEqual(0, score)

    def testWithProcessor(self):
        """
        Any scorer should accept any type as s1 and s2, as long as it is a string
        after preprocessing.
        """
        s1 = ["chicago cubs vs new york mets", "CitiField", "2011-05-11", "8pm"]
        s2 = ["chicago cubs vs new york mets", "CitiFields", "2012-05-11", "9pm"]

        for scorer in scorers:
            score = scorer(s1, s2, processor=lambda event: event[0])
            self.assertEqual(score, 100)

    def testHelp(self):
        """
        test that all help texts can be printed without throwing an exception,
        since they are implemented in C++ aswell
        """

        for scorer in scorers:
            help(scorer)


if __name__ == '__main__':
    unittest.main()