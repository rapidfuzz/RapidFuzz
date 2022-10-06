# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

import warnings
from typing import Callable, Hashable, Sequence

from rapidfuzz.distance import Editop, Hamming, Jaro, JaroWinkler, Levenshtein


def levenshtein(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    weights: tuple[int, int, int] | None = (1, 1, 1),
    processor: Callable[..., Sequence[Hashable]] | None = None,
    max: int | None = None,
    score_cutoff: int | None = None,
) -> int:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Levenshtein.distance instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    score_cutoff = score_cutoff or max
    return Levenshtein.distance(
        s1, s2, weights=weights, processor=processor, score_cutoff=score_cutoff
    )


def levenshtein_editops(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
) -> list[Editop]:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Levenshtein.editops instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Levenshtein.editops(s1, s2, processor=processor).as_list()


def normalized_levenshtein(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    weights: tuple[int, int, int] | None = (1, 1, 1),
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = None,
) -> float:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Levenshtein.normalized_similarity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return (
        Levenshtein.normalized_similarity(
            s1, s2, weights=weights, processor=processor, score_cutoff=score_cutoff
        )
        * 100
    )


def hamming(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    max: int | None = None,
    score_cutoff: int | None = None,
) -> int:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Hamming.distance instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    score_cutoff = score_cutoff or max
    return Hamming.distance(s1, s2, processor=processor, score_cutoff=score_cutoff)


def normalized_hamming(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = None,
) -> float:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Hamming.normalized_similarity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return (
        Hamming.normalized_similarity(
            s1, s2, processor=processor, score_cutoff=score_cutoff
        )
        * 100
    )


def jaro_similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = None,
) -> float:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.Jaro.similarity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Jaro.similarity(s1, s2, processor=processor, score_cutoff=score_cutoff) * 100


def jaro_winkler_similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    prefix_weight: float = 0.1,
    processor: Callable[..., Sequence[Hashable]] | None = None,
    score_cutoff: float | None = None,
) -> float:
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
    warnings.warn(
        "This function will be remove in v3.0.0. Use rapidfuzz.distance.JaroWinkler.similarity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
