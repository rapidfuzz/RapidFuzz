import unittest

from rapidfuzz import process, fuzz, utils

class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.baseball_strings = [
            "new york mets vs chicago cubs",
            "chicago cubs vs chicago white sox",
            "philladelphia phillies vs atlanta braves",
            "braves vs mets",
        ]

    def testGetBestChoice1(self):
        query = "new york mets at atlanta braves"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], "braves vs mets")

    def testGetBestChoice2(self):
        query = "philadelphia phillies at atlanta braves"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[2])

    def testGetBestChoice3(self):
        query = "atlanta braves at philadelphia phillies"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[2])

    def testGetBestChoice4(self):
        query = "chicago cubs vs new york mets"
        best = process.extractOne(query, self.baseball_strings)
        self.assertEqual(best[0], self.baseball_strings[0])

    def testWithProcessor(self):
        events = [
            ["chicago cubs vs new york mets", "CitiField", "2011-05-11", "8pm"],
            ["new york yankees vs boston red sox", "Fenway Park", "2011-05-11", "8pm"],
            ["atlanta braves vs pittsburgh pirates", "PNC Park", "2011-05-11", "8pm"],
        ]
        query = "new york mets vs chicago cubs"

        best = process.extractOne(query, events, processor=lambda event: event[0])
        self.assertEqual(best[0], events[0])

    def testWithScorer(self):
        choices = [
            "new york mets vs chicago cubs",
            "chicago cubs at new york mets",
            "atlanta braves vs pittsbugh pirates",
            "new york yankees vs boston red sox"
        ]

        choices_mapping = {
            1: "new york mets vs chicago cubs",
            2: "chicago cubs at new york mets",
            3: "atlanta braves vs pittsbugh pirates",
            4: "new york yankees vs boston red sox"
        }

        # in this hypothetical example we care about ordering, so we use quick ratio
        query = "new york mets at chicago cubs"

        # first, as an example, the normal way would select the "more 'complete' match of choices[1]"
        best = process.extractOne(query, choices)
        self.assertEqual(best[0], choices[1])
        best = process.extractOne(query, choices_mapping)
        self.assertEqual(best[0], choices_mapping[2])

        # now, use the custom scorer
        best = process.extractOne(query, choices, scorer=fuzz.QRatio)
        self.assertEqual(best[0], choices[0])
        best = process.extractOne(query, choices_mapping, scorer=fuzz.QRatio)
        self.assertEqual(best[0], choices_mapping[1])

    def testWithCutoff(self):
        choices = [
            "new york mets vs chicago cubs",
            "chicago cubs at new york mets",
            "atlanta braves vs pittsbugh pirates",
            "new york yankees vs boston red sox"
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
            "new york yankees vs boston red sox"
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

    def testEmptyStrings(self):
        choices = [
            "",
            "new york mets vs chicago cubs",
            "new york yankees vs boston red sox",
            "",
            ""
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
            None
        ]

        query = "new york mets at chicago cubs"

        best = process.extractOne(query, choices)
        self.assertEqual(best[0], choices[1])

if __name__ == '__main__':
    unittest.main()
