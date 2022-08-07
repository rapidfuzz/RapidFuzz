# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.distance import Levenshtein_py, Hamming_py, Jaro, JaroWinkler


def levenshtein(s1, s2, *, weights=(1, 1, 1), processor=None, max=None):
    """
    Calculates the minimum number of insertions, deletions, and substitutions
    required to change one sequence into the other according to Levenshtein with custom
    costs for insertion, deletion and substitution

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    max : int or None, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        max + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.distance` instead.
        This function will be removed in v3.0.0.

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.string_metric import levenshtein
    >>> levenshtein("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> levenshtein("lewenstein", "levenshtein", max=1)
    2

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    return Levenshtein_py.distance(
        s1, s2, weights=weights, processor=processor, score_cutoff=max
    )


def levenshtein_editops(s1, s2, *, processor=None):
    """
    Return list of 3-tuples describing how to turn s1 into s2.
    Each tuple is of the form (tag, src_pos, dest_pos).

    The tags are strings, with these meanings:
    'replace':  s1[src_pos] should be replaced by s2[dest_pos]
    'delete':   s1[src_pos] should be deleted.
    'insert':   s2[dest_pos] should be inserted at s1[src_pos].

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    editops : list[]
        edit operations required to turn s1 into s2

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.editops` instead.
        This function will be removed in v3.0.0.

    Examples
    --------
    >>> from rapidfuzz.string_metric import levenshtein_editops
    >>> for tag, src_pos, dest_pos in levenshtein_editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[1] s2[0]
    replace s1[3] s2[2]
     insert s1[6] s2[5]
    """
    return Levenshtein_py.editops(s1, s2, processor=processor)


def normalized_levenshtein(
    s1, s2, *, weights=(1, 1, 1), processor=None, score_cutoff=None
):
    """
    Calculates a normalized levenshtein distance using custom
    costs for insertion, deletion and substitution.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.normalized_similarity` instead.
        This function will be removed in v3.0.0.

    See Also
    --------
    levenshtein : Levenshtein distance

    Examples
    --------
    Find the normalized Levenshtein distance between two strings:

    >>> from rapidfuzz.string_metric import normalized_levenshtein
    >>> normalized_levenshtein("lewenstein", "levenshtein")
    81.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> normalized_levenshtein("lewenstein", "levenshtein", score_cutoff=85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> normalized_levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
    85.71428571428571

     When a different processor is used s1 and s2 do not have to be strings

    >>> normalized_levenshtein(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    81.81818181818181
    """
    return (
        Levenshtein_py.normalized_similarity(
            s1, s2, weights=weights, processor=processor, score_cutoff=score_cutoff
        )
        * 100
    )


def hamming(s1, s2, *, processor=None, max=None):
    """
    Calculates the Hamming distance between two strings.
    The hamming distance is defined as the number of positions
    where the two strings differ. It describes the minimum
    amount of substitutions required to transform s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    max : int or None, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        max + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If s1 and s2 have a different length

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Hamming.distance` instead.
        This function will be removed in v3.0.0.
    """
    return Hamming_py.distance(s1, s2, processor=processor, score_cutoff=max)


def normalized_hamming(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized hamming distance

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If s1 and s2 have a different length

    See Also
    --------
    hamming : Hamming distance

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Hamming.normalized_similarity` instead.
        This function will be removed in v3.0.0.
    """
    return (
        Hamming_py.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        * 100
    )


def jaro_similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the jaro similarity

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Jaro.similarity` instead.
        This function will be removed in v3.0.0.
    """
    return Jaro.similarity(s1, s2, processor=processor, score_cutoff=score_cutoff) * 100


def jaro_winkler_similarity(
    s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None
):
    """
    Calculates the jaro winkler similarity

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    prefix_weight : float, optional
        Weight used for the common prefix of the two strings.
        Has to be between 0 and 0.25. Default is 0.1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If prefix_weight is invalid

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.JaroWinkler.similarity` instead.
        This function will be removed in v3.0.0.
    """
    return (
        JaroWinkler.similarity(
            s1,
            s2,
            prefix_weight=prefix_weight,
            processor=processor,
            score_cutoff=score_cutoff,
        )
        * 100
    )


def _GetScorerFlagsDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 2**63 - 1, "flags": (1 << 6)}


def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 100, "worst_score": 0, "flags": (1 << 5)}


levenshtein._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsDistance}

normalized_levenshtein._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

hamming._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsDistance}

normalized_hamming._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

jaro_similarity._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

jaro_winkler_similarity._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}
