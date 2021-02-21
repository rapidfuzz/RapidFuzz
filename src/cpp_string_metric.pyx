# distutils: language=c++
# cython: language_level=3
# cython: binding=True

from rapidfuzz.utils import default_process

cdef extern from "cpp_string_metric.hpp":
    object levenshtein_impl(object, object, size_t, size_t, size_t, size_t) except +*
    double normalized_levenshtein_impl(object, object, size_t, size_t, size_t, double) except +*
    double normalized_levenshtein_impl_default_process(object, object, size_t, size_t, size_t, double) except +*

    object hamming_impl(object, object, size_t) except +*
    double normalized_hamming_impl(object, object, double) except +*
    double normalized_hamming_impl_default_process(object, object, double) except +*


cdef dummy() except +:
    # trick cython into generating
    # exception handling, since except +* does not work properly
    # https://github.com/cython/cython/issues/3065
    dummy()


def levenshtein(s1, s2, weights=(1,1,1), max=None):
    """
    Calculates the minimum number of insertions, deletions, and substitutions
    required to change one sequence into the other according to Levenshtein with custom
    costs for insertion, deletion and substitution

    Parameters
    ----------
    s1 : str
        First string to compare
    s2 : str
        Second string to compare
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    max : int or None, optional
        Maximum Levenshtein distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        -1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        levenshtein distance between s1 and s2

    Notes
    -----
    Depending on the input parameters different optimized implementation are used
    to improve the performance. Worst-case performance is ``O(m * n)``.

    Insertion = 1, Deletion = 1, Substitution = 1:
      - if max is 0 the similarity can be calculated using a direct comparision,
        since no difference between the strings is allowed.  The time complexity of
        this algorithm is ``O(N)``.

      - A common prefix/suffix of the two compared strings does not affect
        the Levenshtein distance, so the affix is removed before calculating the
        similarity.

      - If max is ≤ 3 the mbleven algorithm is used. This algorithm
        checks all possible edit operations that are possible under
        the threshold `max`. The time complexity of this algorithm is ``O(N)``.

      - If the length of the shorter string is ≤ 64 after removing the common affix
        Hyyrös' algorithm is used, which calculates the Levenshtein distance in
        parallel. The algorithm is described by [1]_. The time complexity of this
        algorithm is ``O(N)``.

      - If the length of the shorter string is ≥ 64 after removing the common affix
        a blockwise implementation of Myers' algorithm is used, which calculates
        the Levenshtein distance in parallel (64 characters at a time).
        The algorithm is described by [3]_. The time complexity of this
        algorithm is ``O([N/64]M)``.


    Insertion = 1, Deletion = 1, Substitution >= Insertion + Deletion:
      when ``Substitution >= Insertion + Deletion`` set
      ``Substitution = Insertion + Deletion``
      since every Substitution can be performed as Insertion + Deletion
      so in this case treat Substitution as 2

      - if max is 0 the similarity can be calculated using a direct comparision,
        since no difference between the strings is allowed.  The time complexity of
        this algorithm is ``O(N)``.

      - if max is 1 and the two strings have a similar length, the similarity can be
        calculated using a direct comparision aswell, since a substitution would cause
        a edit distance higher than max. The time complexity of this algorithm
        is ``O(N)``.

      - A common prefix/suffix of the two compared strings does not affect
        the Levenshtein distance, so the affix is removed before calculating the
        similarity.

      - If max is ≤ 4 the mbleven algorithm is used. This algorithm
        checks all possible edit operations that are possible under
        the threshold `max`. As a difference to the normal Levenshtein distance this
        algorithm can even be used up to a threshold of 4 here, since the higher weight
        of substitutions decreases the amount of possible edit operations.
        The time complexity of this algorithm is ``O(N)``.

      - If the length of the shorter string is ≤ 64 after removing the common affix
        the BitPAl algorithm is used, which calculates the Levenshtein distance in
        parallel. The algorithm is described by [4]_ and is extended with support
        for UTF32 in this implementation. The time complexity of this
        algorithm is ``O(N)``.

      - In all other cases the Levenshtein distance is calculated using
        Wagner-Fischer with Ukkonens optimization as described by [2]_. The time
        complexity of this algorithm is ``O(N * M)``.
        This can be replaced with a blockwise implementation of the BitPal algorithm
        in the future.

    Other weights:
      The implementation for other weights is based on Wagner-Fischer.
      It has a performance of ``O(N * M)`` and has a memory usage of ``O(N)``.
      Further details can be found in [2]_.


    References
    ----------
    .. [1] Hyyrö, Heikki. "A Bit-Vector Algorithm for Computing
           Levenshtein and Damerau Edit Distances."
           Nordic Journal of Computing, Volume 10 (2003): 29-39.
    .. [2] Wagner, Robert & Fischer, Michael
           "The String-to-String Correction Problem."
           J. ACM. 21. (1974): 168-173
    .. [3] Myers, Gene. "A fast bit-vector algorithm for approximate
           string matching based on dynamic programming."
           Journal of the ACM (JACM) 46.3 (1999): 395-415.
    .. [4] Loving, Joshua & Hernández, Yözen & Benson, Gary.
           "BitPAl: A Bit-Parallel, General Integer-Scoring Sequence
           Alignment Algorithm. Bioinformatics"
           Bioinformatics, Volume 30 (2014): 3166–3173

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.string_metric import levenshtein
    >>> levenshtein("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> levenshtein("lewenstein", "levenshtein", max=1)
    -1

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    cdef size_t insertion = 1
    cdef size_t deletion = 1
    cdef size_t substitution = 1
    cdef size_t max_ = -1

    if weights:
        insertion, deletion, substitution = weights

    if max is not None:
        max_ = max

    return levenshtein_impl(s1, s2, insertion, deletion, substitution, max_)


def normalized_levenshtein(s1, s2, weights=(1,1,1), processor=None, double score_cutoff=0.0):
    """
    Calculates a normalized levenshtein distance using custom
    costs for insertion, deletion and substitution.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: bool or callable, optional
      Optional callable that is used to preprocess the strings before
      comparing them. When processor is True ``utils.default_process``
      is used. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        Normalized weighted levenshtein distance between s1 and s2
        as a float between 0 and 100

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    See Also
    --------
    levenshtein : Levenshtein distance

    Notes
    -----
    The normalization of the Levenshtein distance is performed in the following way:

    .. math::
      :nowrap:

      \\begin{align*}
        dist_{max} &= \\begin{cases}
          min(len(s1), len(s2)) \cdot sub,       & \\text{if } sub \leq ins + del \\\\
          len(s1) \cdot del + len(s2) \cdot ins, & \\text{otherwise}
        \end{cases}\\\\[10pt]

        dist_{max} &= \\begin{cases}
          dist_{max} + (len(s1) - len(s2)) \cdot del, & \\text{if } len(s1) > len(s2) \\\\
          dist_{max} + (len(s2) - len(s1)) \cdot ins, & \\text{if } len(s1) < len(s2) \\\\
          dist_{max},                                 & \\text{if } len(s1) = len(s2)
        \end{cases}\\\\[10pt]

        ratio &= 100 \cdot \\frac{distance(s1, s2)}{dist_{max}}
      \end{align*}

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
    cdef size_t insertion = 1
    cdef size_t deletion = 1
    cdef size_t substitution = 1

    if s1 is None or s2 is None:
        return 0

    if weights:
        insertion, deletion, substitution = weights

    if processor is True or processor == default_process:
        return normalized_levenshtein_impl_default_process(
            s1, s2, insertion, deletion, substitution, score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return normalized_levenshtein_impl(s1, s2, insertion, deletion, substitution, score_cutoff)


def hamming(s1, s2, max=None):
    """
    Calculates the Hamming distance between two strings.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    max : int or None, optional
        Maximum Hamming distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        -1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        Hamming distance between s1 and s2
    """
    cdef size_t max_ = -1

    if max is not None:
        max_ = max

    return hamming_impl(s1, s2, max_)


def normalized_hamming(s1, s2, processor=None, double score_cutoff=0.0):
    """
    Calculates a normalized hamming distance

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
      Optional callable that is used to preprocess the strings before
      comparing them. When processor is True ``utils.default_process``
      is used. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        Normalized hamming distance between s1 and s2
        as a float between 0 and 100
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return normalized_hamming_impl_default_process(s1, s2, score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return normalized_hamming_impl(s1, s2, score_cutoff)
