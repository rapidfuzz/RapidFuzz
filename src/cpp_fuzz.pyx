# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.utils import default_process
from cpp_common cimport RfString, RfStringWrapper, is_valid_string, convert_string, hash_array, hash_sequence
from array import array
from libcpp.utility cimport move

cdef inline RfString conv_sequence(seq) except *:
    if is_valid_string(seq):
        return move(convert_string(seq))
    elif isinstance(seq, array):
        return move(hash_array(seq))
    else:
        return move(hash_sequence(seq))

cdef extern from "cpp_scorer.hpp":
    double ratio_no_process(                         const RfString&, const RfString&, double) nogil except +
    double ratio_default_process(                    const RfString&, const RfString&, double) nogil except +
    double partial_ratio_no_process(                 const RfString&, const RfString&, double) nogil except +
    double partial_ratio_default_process(            const RfString&, const RfString&, double) nogil except +
    double token_sort_ratio_no_process(              const RfString&, const RfString&, double) nogil except +
    double token_sort_ratio_default_process(         const RfString&, const RfString&, double) nogil except +
    double token_set_ratio_no_process(               const RfString&, const RfString&, double) nogil except +
    double token_set_ratio_default_process(          const RfString&, const RfString&, double) nogil except +
    double token_ratio_no_process(                   const RfString&, const RfString&, double) nogil except +
    double token_ratio_default_process(              const RfString&, const RfString&, double) nogil except +
    double partial_token_sort_ratio_no_process(      const RfString&, const RfString&, double) nogil except +
    double partial_token_sort_ratio_default_process( const RfString&, const RfString&, double) nogil except +
    double partial_token_set_ratio_no_process(       const RfString&, const RfString&, double) nogil except +
    double partial_token_set_ratio_default_process(  const RfString&, const RfString&, double) nogil except +
    double partial_token_ratio_no_process(           const RfString&, const RfString&, double) nogil except +
    double partial_token_ratio_default_process(      const RfString&, const RfString&, double) nogil except +
    double WRatio_no_process(                        const RfString&, const RfString&, double) nogil except +
    double WRatio_default_process(                   const RfString&, const RfString&, double) nogil except +
    double QRatio_no_process(                        const RfString&, const RfString&, double) nogil except +
    double QRatio_default_process(                   const RfString&, const RfString&, double) nogil except +

def ratio(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the normalized InDel distance.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
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
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    See Also
    --------
    rapidfuzz.string_metric.normalized_levenshtein : Normalized levenshtein distance

    Notes
    -----
    .. image:: img/ratio.svg

    Examples
    --------
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0
    
    if processor is True: #: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))

    return ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio(s1, s2, *, processor=None, score_cutoff=None):
    """
    Searches for the optimal alignment of the shorter string in the
    longer string and returns the fuzz.ratio for this alignment.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
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
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    Depending on the length of the needle (shorter string) different
    implementations are used to improve the performance.

    short needle (length ≤ 64):
        When using a short needle length the fuzz.ratio is calculated for all
        alignments that could result in an optimal alignment. It is
        guaranteed to find the optimal alignment. For short needles this is very
        fast, since for them fuzz.ratio runs in ``O(N)`` time. This results in a worst
        case performance of ``O(NM)``.
    
    .. image:: img/partial_ratio_short_needle.svg

    long needle (length > 64):
        For long needles a similar implementation to FuzzyWuzzy is used.
        This implementation only considers alignments which start at one
        of the longest common substrings. This results in a worst case performance
        of ``O(N[N/64]M)``. However usually most of the alignments can be skipped.
        The following Python code shows the concept:

        .. code-block:: python

            blocks = SequenceMatcher(None, needle, longer, False).get_matching_blocks()
            score = 0
            for block in blocks:
                long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
                long_end = long_start + len(shorter)
                long_substr = longer[long_start:long_end]
                score = max(score, fuzz.ratio(needle, long_substr))

        This is a lot faster than checking all possible alignments. However it
        only finds one of the best alignments and not necessarily the optimal one.

    .. image:: img/partial_ratio_long_needle.svg

    Examples
    --------
    >>> fuzz.partial_ratio("this is a test", "this is a test!")
    100.0
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return partial_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return partial_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_sort_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Sorts the words in the strings and calculates the fuzz.ratio between them

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_sort_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100.0
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return token_sort_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return token_sort_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_set_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.ratio

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_set_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    83.8709716796875
    >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100.0
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return token_set_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return token_set_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
    (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_ratio.svg
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return token_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return token_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_sort_ratio.svg
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return partial_token_sort_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return partial_token_sort_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.partial_ratio

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_set_ratio.svg
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return partial_token_set_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return partial_token_set_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_ratio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and
    fuzz.partial_token_sort_ratio (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_ratio.svg
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return partial_token_ratio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return partial_token_ratio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def WRatio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/WRatio.svg
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return WRatio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return WRatio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)


def QRatio(s1, s2, *, processor=True, score_cutoff=None):
    """
    Calculates a quick ratio between two strings using fuzz.ratio.
    The only difference to fuzz.ratio is, that this preprocesses
    the strings by default.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Examples
    --------
    >>> fuzz.QRatio("this is a test", "THIS is a test!")
    100.0
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if s1 is None or s2 is None:
        return 0

    if processor is True: #or processor == default_process:
        s1_proc = RfStringWrapper(conv_sequence(s1))
        s2_proc = RfStringWrapper(conv_sequence(s2))
        return QRatio_default_process(s1_proc.string, s2_proc.string, c_score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    s1_proc = RfStringWrapper(conv_sequence(s1))
    s2_proc = RfStringWrapper(conv_sequence(s2))
    return QRatio_no_process(s1_proc.string, s2_proc.string, c_score_cutoff)