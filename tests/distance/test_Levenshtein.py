from rapidfuzz import process
from rapidfuzz.distance import Levenshtein_cpp, Levenshtein_py, Opcode, Opcodes
from ..common import GenericScorer


class CustomHashable:
    def __init__(self, string):
        self._string = string

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._string)


def get_scorer_flags(s1, s2, weights=(1, 1, 1), **kwargs):
    insert_cost, delete_cost, replace_cost = weights
    max_dist = len(s1) * delete_cost + len(s2) * insert_cost

    if len(s1) >= len(s2):
        max_dist = min(
            max_dist, len(s2) * replace_cost + (len(s1) - len(s2)) * delete_cost
        )
    else:
        max_dist = min(
            max_dist, len(s1) * replace_cost + (len(s2) - len(s1)) * insert_cost
        )

    return {"maximum": max_dist, "symmetric": insert_cost == delete_cost}


Levenshtein = GenericScorer(Levenshtein_py, Levenshtein_cpp, get_scorer_flags)


def test_empty_string():
    """
    when both strings are empty this is a perfect match
    """
    assert Levenshtein.distance("", "") == 0
    assert Levenshtein.distance("", "", weights=(1, 1, 0)) == 0
    assert Levenshtein.distance("", "", weights=(1, 1, 2)) == 0
    assert Levenshtein.distance("", "", weights=(1, 1, 5)) == 0
    assert Levenshtein.distance("", "", weights=(3, 7, 5)) == 0


def test_cross_type_matching():
    """
    strings should always be interpreted in the same way
    """
    assert Levenshtein.distance("aaaa", "aaaa") == 0
    assert Levenshtein.distance("aaaa", ["a", "a", "a", "a"]) == 0
    # todo add support in pure python
    assert Levenshtein_cpp.distance("aaaa", [ord("a"), ord("a"), "a", "a"]) == 0
    assert Levenshtein_cpp.distance([0, -1], [0, -2]) == 1
    assert (
        Levenshtein_cpp.distance(
            [CustomHashable("aa"), CustomHashable("aa")],
            [CustomHashable("aa"), CustomHashable("bb")],
        )
        == 1
    )


def test_word_error_rate():
    """
    it should be possible to use levenshtein to implement a word error rate
    """
    assert Levenshtein.distance(["aaaaa", "bbbb"], ["aaaaa", "bbbb"]) == 0
    assert Levenshtein.distance(["aaaaa", "bbbb"], ["aaaaa", "cccc"]) == 1


def test_simple_unicode_tests():
    """
    some very simple tests using unicode with scorers
    to catch relatively obvious implementation errors
    """
    s1 = "ÁÄ"
    s2 = "ABCD"
    assert Levenshtein.distance(s1, s2) == 4  # 2 sub + 2 ins
    assert Levenshtein.distance(s1, s2, weights=(1, 1, 0)) == 2  # 2 sub + 2 ins
    assert (
        Levenshtein.distance(s1, s2, weights=(1, 1, 2)) == 6
    )  # 2 del + 4 ins / 2 sub + 2 ins
    assert Levenshtein.distance(s1, s2, weights=(1, 1, 5)) == 6  # 2 del + 4 ins
    assert Levenshtein.distance(s1, s2, weights=(1, 7, 5)) == 12  # 2 sub + 2 ins
    assert Levenshtein.distance(s2, s1, weights=(1, 7, 5)) == 24  # 2 sub + 2 del

    assert Levenshtein.distance(s1, s1) == 0
    assert Levenshtein.distance(s1, s1, weights=(1, 1, 0)) == 0
    assert Levenshtein.distance(s1, s1, weights=(1, 1, 2)) == 0
    assert Levenshtein.distance(s1, s1, weights=(1, 1, 5)) == 0
    assert Levenshtein.distance(s1, s1, weights=(3, 7, 5)) == 0


def test_Editops():
    """
    basic test for Levenshtein_cpp.editops
    """
    assert Levenshtein_cpp.editops("0", "").as_list() == [("delete", 0, 0)]
    assert Levenshtein_cpp.editops("", "0").as_list() == [("insert", 0, 0)]

    assert Levenshtein_cpp.editops("00", "0").as_list() == [("delete", 1, 1)]
    assert Levenshtein_cpp.editops("0", "00").as_list() == [("insert", 1, 1)]

    assert Levenshtein_cpp.editops("qabxcd", "abycdf").as_list() == [
        ("delete", 0, 0),
        ("replace", 3, 2),
        ("insert", 6, 5),
    ]
    assert Levenshtein_cpp.editops("Lorem ipsum.", "XYZLorem ABC iPsum").as_list() == [
        ("insert", 0, 0),
        ("insert", 0, 1),
        ("insert", 0, 2),
        ("insert", 6, 9),
        ("insert", 6, 10),
        ("insert", 6, 11),
        ("insert", 6, 12),
        ("replace", 7, 14),
        ("delete", 11, 18),
    ]

    ops = Levenshtein_cpp.editops("aaabaaa", "abbaaabba")
    assert ops.src_len == 7
    assert ops.dest_len == 9


def test_Opcodes():
    """
    basic test for Levenshtein_cpp.opcodes
    """
    assert Levenshtein_cpp.opcodes("", "abc") == Opcodes(
        [Opcode(tag="insert", src_start=0, src_end=0, dest_start=0, dest_end=3)], 0, 3
    )


def test_mbleven():
    """
    test for regressions to previous bugs in the cached Levenshtein implementation
    """
    assert 2 == Levenshtein.distance("0", "101", score_cutoff=1)
    assert 2 == Levenshtein.distance("0", "101", score_cutoff=2)
    assert 2 == Levenshtein.distance("0", "101", score_cutoff=3)

    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_cpp.distance, processor=None, score_cutoff=1
    )
    assert match is None
    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_py.distance, processor=None, score_cutoff=1
    )
    assert match is None
    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_cpp.distance, processor=None, score_cutoff=2
    )
    assert match == ("101", 2, 0)
    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_py.distance, processor=None, score_cutoff=2
    )
    assert match == ("101", 2, 0)
    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_cpp.distance, processor=None, score_cutoff=3
    )
    assert match == ("101", 2, 0)
    match = process.extractOne(
        "0", ["101"], scorer=Levenshtein_py.distance, processor=None, score_cutoff=3
    )
    assert match == ("101", 2, 0)
