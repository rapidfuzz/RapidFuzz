from rapidfuzz.distance import (
    DamerauLevenshtein_cpp,
    DamerauLevenshtein_py,
    Hamming_cpp,
    Hamming_py,
    Indel_cpp,
    Indel_py,
    Jaro_cpp,
    Jaro_py,
    JaroWinkler_cpp,
    JaroWinkler_py,
    LCSseq_cpp,
    LCSseq_py,
    Levenshtein_cpp,
    Levenshtein_py,
    OSA_cpp,
    OSA_py,
    Postfix_cpp,
    Postfix_py,
    Prefix_cpp,
    Prefix_py,
)
from tests.common import GenericScorer, is_none


def get_scorer_flags_damerau_levenshtein(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


DamerauLevenshtein = GenericScorer(
    DamerauLevenshtein_py, DamerauLevenshtein_cpp, get_scorer_flags_damerau_levenshtein
)


def get_scorer_flags_hamming(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


Hamming = GenericScorer(Hamming_py, Hamming_cpp, get_scorer_flags_hamming)


def get_scorer_flags_indel(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": len(s1) + len(s2), "symmetric": True}


Indel = GenericScorer(Indel_py, Indel_cpp, get_scorer_flags_indel)


def get_scorer_flags_jaro(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": 1.0, "symmetric": True}


Jaro = GenericScorer(Jaro_py, Jaro_cpp, get_scorer_flags_jaro)


def get_scorer_flags_jaro_winkler(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": 1.0, "symmetric": True}


JaroWinkler = GenericScorer(
    JaroWinkler_py, JaroWinkler_cpp, get_scorer_flags_jaro_winkler
)


def get_scorer_flags_lcs_seq(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


LCSseq = GenericScorer(LCSseq_py, LCSseq_cpp, get_scorer_flags_lcs_seq)


def get_scorer_flags_levenshtein(s1, s2, weights=(1, 1, 1), **kwargs):
    insert_cost, delete_cost, replace_cost = weights

    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": insert_cost == delete_cost}

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


Levenshtein = GenericScorer(
    Levenshtein_py, Levenshtein_cpp, get_scorer_flags_levenshtein
)


def get_scorer_flags_osa(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


OSA = GenericScorer(OSA_py, OSA_cpp, get_scorer_flags_osa)


def get_scorer_flags_postfix(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


Postfix = GenericScorer(Postfix_py, Postfix_cpp, get_scorer_flags_postfix)


def get_scorer_flags_prefix(s1, s2, **kwargs):
    if is_none(s1) or is_none(s2):
        return {"maximum": None, "symmetric": True}
    return {"maximum": max(len(s1), len(s2)), "symmetric": True}


Prefix = GenericScorer(Prefix_py, Prefix_cpp, get_scorer_flags_prefix)

all_scorer_modules = [
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    OSA,
    Postfix,
    Prefix,
]
