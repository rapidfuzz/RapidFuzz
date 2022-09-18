#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from rapidfuzz import process_py, process_cpp, fuzz


class process:
    @staticmethod
    def extract_iter(*args, **kwargs):
        res1 = process_cpp.extract_iter(*args, **kwargs)
        res2 = process_py.extract_iter(*args, **kwargs)

        for elem1, elem2 in zip(res1, res2, strict=True):
            assert elem1 == elem2
            yield elem1

    @staticmethod
    def extractOne(*args, **kwargs):
        res1 = process_cpp.extractOne(*args, **kwargs)
        res2 = process_py.extractOne(*args, **kwargs)
        assert res1 == res2
        return res1

    @staticmethod
    def extract(*args, **kwargs):
        res1 = process_cpp.extract(*args, **kwargs)
        res2 = process_py.extract(*args, **kwargs)
        assert res1 == res2
        return res1


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.baseball_strings = [
            "new york mets vs chicago cubs",
            "chicago cubs vs chicago white sox",
            "philladelphia phillies vs atlanta braves",
            "braves vs mets",
        ]

    def testExtractOneExceptions(self):
        self.assertRaises(TypeError, process_cpp.extractOne)
        self.assertRaises(TypeError, process_py.extractOne)
        self.assertRaises(TypeError, process_cpp.extractOne, 1)
        self.assertRaises(TypeError, process_py.extractOne, 1)
        self.assertRaises(TypeError, process_cpp.extractOne, 1, [])
        self.assertRaises(TypeError, process_py.extractOne, 1, [])
        self.assertRaises(TypeError, process_cpp.extractOne, "", [1])
        self.assertRaises(TypeError, process_py.extractOne, "", [1])
        self.assertRaises(TypeError, process_cpp.extractOne, "", {1: 1})
        self.assertRaises(TypeError, process_py.extractOne, "", {1: 1})

    def testExtractExceptions(self):
        self.assertRaises(TypeError, process_cpp.extract)
        self.assertRaises(TypeError, process_py.extract)
        self.assertRaises(TypeError, process_cpp.extract, 1)
        self.assertRaises(TypeError, process_py.extract, 1)
        self.assertRaises(TypeError, process_cpp.extract, 1, [])
        self.assertRaises(TypeError, process_py.extract, 1, [])
        self.assertRaises(TypeError, process_cpp.extract, "", [1])
        self.assertRaises(TypeError, process_py.extract, "", [1])
        self.assertRaises(TypeError, process_cpp.extract, "", {1: 1})
        self.assertRaises(TypeError, process_py.extract, "", {1: 1})

    def testExtractIterExceptions(self):
        self.assertRaises(TypeError, process_cpp.extract_iter)
        self.assertRaises(TypeError, process_py.extract_iter)
        self.assertRaises(TypeError, process_cpp.extract_iter, 1)
        self.assertRaises(TypeError, process_py.extract_iter, 1)
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_cpp.extract_iter(*args, **kwargs)),
            1,
            [],
        )
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_py.extract_iter(*args, **kwargs)),
            1,
            [],
        )
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_cpp.extract_iter(*args, **kwargs)),
            "",
            [1],
        )
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_py.extract_iter(*args, **kwargs)),
            "",
            [1],
        )
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_cpp.extract_iter(*args, **kwargs)),
            "",
            {1: 1},
        )
        self.assertRaises(
            TypeError,
            lambda *args, **kwargs: next(process_py.extract_iter(*args, **kwargs)),
            "",
            {1: 1},
        )

    def testGetBestChoice1(self):
        query = "new york mets at atlanta braves"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], "braves vs mets")
        best = process.extractOne(query, set(self.baseball_strings))
        self.assertEqual(best[0], "braves vs mets")

        best = process.extract(query, self.baseball_strings)[0]
        self.assertEqual(best[0], "braves vs mets")
        best = process.extract(query, set(self.baseball_strings))[0]
        self.assertEqual(best[0], "braves vs mets")

    def testGetBestChoice2(self):
        query = "philadelphia phillies at atlanta braves"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[2])
        best = process.extractOne(query, set(self.baseball_strings))
        self.assertEqual(best[0], self.baseball_strings[2])

        best = process.extract(query, self.baseball_strings)[0]
        self.assertEqual(best[0], self.baseball_strings[2])
        best = process.extract(query, set(self.baseball_strings))[0]
        self.assertEqual(best[0], self.baseball_strings[2])

    def testGetBestChoice3(self):
        query = "atlanta braves at philadelphia phillies"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[2])
        best = process.extractOne(query, set(self.baseball_strings))
        self.assertEqual(best[0], self.baseball_strings[2])

        best = process.extract(query, self.baseball_strings)[0]
        self.assertEqual(best[0], self.baseball_strings[2])
        best = process.extract(query, set(self.baseball_strings))[0]
        self.assertEqual(best[0], self.baseball_strings[2])

    def testGetBestChoice4(self):
        query = "chicago cubs vs new york mets"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[0])
        best = process.extractOne(query, set(self.baseball_strings))
        self.assertEqual(best[0], self.baseball_strings[0])

    def testWithProcessor(self):
        """
        extractOne should accept any type as long as it is a string
        after preprocessing
        """
        events = [
            ["chicago cubs vs new york mets", "CitiField", "2011-05-11", "8pm"],
            ["new york yankees vs boston red sox", "Fenway Park", "2011-05-11", "8pm"],
            ["atlanta braves vs pittsburgh pirates", "PNC Park", "2011-05-11", "8pm"],
        ]
        query = events[0]

        best = process.extractOne(query, events, processor=lambda event: event[0])
        self.assertEqual(best[0], events[0])

    def testWithScorer(self):
        choices = [
            "new york mets vs chicago cubs",
            "chicago cubs at new york mets",
            "atlanta braves vs pittsbugh pirates",
            "new york yankees vs boston red sox",
        ]

        choices_mapping = {
            1: "new york mets vs chicago cubs",
            2: "chicago cubs at new york mets",
            3: "atlanta braves vs pittsbugh pirates",
            4: "new york yankees vs boston red sox",
        }

        # in this hypothetical example we care about ordering, so we use quick ratio
        query = "new york mets at chicago cubs"

        # first, as an example, the normal way would select the "more 'complete' match of choices[1]"
        best = process.extractOne(query, choices)
        self.assertEqual(best[0], choices[1])
        best = process.extract(query, choices)[0]
        self.assertEqual(best[0], choices[1])
        # dict
        best = process.extractOne(query, choices_mapping)
        self.assertEqual(best[0], choices_mapping[2])
        best = process.extract(query, choices_mapping)[0]
        self.assertEqual(best[0], choices_mapping[2])

        # now, use the custom scorer
        best = process.extractOne(query, choices, scorer=fuzz.QRatio)
        self.assertEqual(best[0], choices[0])
        best = process.extract(query, choices, scorer=fuzz.QRatio)[0]
        self.assertEqual(best[0], choices[0])
        # dict
        best = process.extractOne(query, choices_mapping, scorer=fuzz.QRatio)
        self.assertEqual(best[0], choices_mapping[1])
        best = process.extract(query, choices_mapping, scorer=fuzz.QRatio)[0]
        self.assertEqual(best[0], choices_mapping[1])

    def testWithCutoff(self):
        choices = [
            "new york mets vs chicago cubs",
            "chicago cubs at new york mets",
            "atlanta braves vs pittsbugh pirates",
            "new york yankees vs boston red sox",
        ]

        query = "los angeles dodgers vs san francisco giants"

        # in this situation, this is an event that does not exist in the list
        # we don't want to randomly match to something, so we use a reasonable cutoff
        best = process.extractOne(query, choices, score_cutoff=50)
        self.assertIsNone(best)

        # however if we had no cutoff, something would get returned
        best = process.extractOne(query, choices)
        self.assertIsNotNone(best)

    def testWithCutoffEdgeCases(self):
        choices = [
            "new york mets vs chicago cubs",
            "chicago cubs at new york mets",
            "atlanta braves vs pittsbugh pirates",
            "new york yankees vs boston red sox",
        ]

        query = "new york mets vs chicago cubs"
        # Only find 100-score cases
        best = process.extractOne(query, choices, score_cutoff=100)
        self.assertIsNotNone(best)
        self.assertEqual(best[0], choices[0])

        # 0-score cases do not return None
        best = process.extractOne("", choices)
        self.assertIsNotNone(best)
        self.assertEqual(best[1], 0)

    def testNoneElements(self):
        """
        when a None element is used, it is skipped and the index is still correct
        """
        best = process.extractOne("test", [None, "tes"])
        self.assertEqual(best[2], 1)
        best = process.extractOne(None, [None, "tes"])
        self.assertEqual(best, None)

        best = process.extract("test", [None, "tes"])
        self.assertEqual(best[0][2], 1)
        best = process.extract(None, [None, "tes"])
        self.assertEqual(best, [])

    def testResultOrder(self):
        """
        when multiple elements have the same score, the first one should be returned
        """
        best = process.extractOne("test", ["tes", "tes"])
        self.assertEqual(best[2], 0)

        best = process.extract("test", ["tes", "tes"], limit=1)
        self.assertEqual(best[0][2], 0)

    def testEmptyStrings(self):
        choices = [
            "",
            "new york mets vs chicago cubs",
            "new york yankees vs boston red sox",
            "",
            "",
        ]

        query = "new york mets at chicago cubs"

        best = process.extractOne(query, choices)
        self.assertEqual(best[0], choices[1])

    def testNullStrings(self):
        choices = [
            None,
            "new york mets vs chicago cubs",
            "new york yankees vs boston red sox",
            None,
            None,
        ]

        query = "new york mets at chicago cubs"

        best = process.extractOne(query, choices)
        self.assertEqual(best[0], choices[1])

    def testIssue81(self):
        # this mostly tests whether this segfaults due to incorrect ref counting
        pd = pytest.importorskip("pandas")
        choices = pd.Series(
            ["test color brightness", "test lemon", "test lavender"],
            index=[67478, 67479, 67480],
        )
        matches = process.extract("test", choices)
        assert matches == [
            ("test color brightness", 90.0, 67478),
            ("test lemon", 90.0, 67479),
            ("test lavender", 90.0, 67480),
        ]


def custom_scorer(s1, s2, processor=None, score_cutoff=0):
    return fuzz.ratio(s1, s2, processor=processor, score_cutoff=score_cutoff)


@pytest.mark.parametrize("processor", [False, None, lambda s: s])
@pytest.mark.parametrize("scorer", [fuzz.ratio, custom_scorer])
def test_extractOne_case_sensitive(processor, scorer):
    assert (
        process.extractOne(
            "new york mets",
            ["new", "new YORK mets"],
            processor=processor,
            scorer=scorer,
        )[1]
        != 100
    )


@pytest.mark.parametrize("scorer", [fuzz.ratio, custom_scorer])
def test_extractOne_use_first_match(scorer):
    assert (
        process.extractOne(
            "new york mets", ["new york mets", "new york mets"], scorer=scorer
        )[2]
        == 0
    )


if __name__ == "__main__":
    unittest.main()
