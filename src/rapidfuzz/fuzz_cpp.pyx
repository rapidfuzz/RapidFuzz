# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from .distance._initialize_cpp import ScoreAlignment

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_ScorerFunc, RF_Kwargs, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_SYMMETRIC
)

# required for preprocess_strings
from rapidfuzz.utils import default_process
from array import array
from cpp_common cimport (
    RF_StringWrapper, preprocess_strings, RfScoreAlignment, NoKwargsInit,
    CreateScorerContext, CreateScorerContextPy, AddScorerContext
)

from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cython.operator cimport dereference

from array import array

cdef extern from "fuzz_cpp.hpp":
    double ratio_func(                    const RF_String&, const RF_String&, double) nogil except +
    double partial_ratio_func(            const RF_String&, const RF_String&, double) nogil except +
    double token_sort_ratio_func(         const RF_String&, const RF_String&, double) nogil except +
    double token_set_ratio_func(          const RF_String&, const RF_String&, double) nogil except +
    double token_ratio_func(              const RF_String&, const RF_String&, double) nogil except +
    double partial_token_sort_ratio_func( const RF_String&, const RF_String&, double) nogil except +
    double partial_token_set_ratio_func(  const RF_String&, const RF_String&, double) nogil except +
    double partial_token_ratio_func(      const RF_String&, const RF_String&, double) nogil except +
    double WRatio_func(                   const RF_String&, const RF_String&, double) nogil except +
    double QRatio_func(                   const RF_String&, const RF_String&, double) nogil except +

    RfScoreAlignment[double] partial_ratio_alignment_func(const RF_String&, const RF_String&, double) nogil except +

    bool RatioInit(                 RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialRatioInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenSortRatioInit(        RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenSetRatioInit(         RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenRatioInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenSortRatioInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenSetRatioInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenRatioInit(     RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool WRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool QRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def ratio(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the normalized Indel distance.

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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio_alignment(s1, s2, *, processor=None, score_cutoff=None):
    """
    Searches for the optimal alignment of the shorter string in the
    longer string and returns the fuzz.ratio and the corresponding
    alignment.

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
        For ratio < score_cutoff None is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    alignment : ScoreAlignment, optional
        alignment between s1 and s2 with the score as a float between 0 and 100

    Examples
    --------
    >>> s1 = "a certain string"
    >>> s2 = "cetain"
    >>> res = fuzz.partial_ratio_alignment(s1, s2)
    >>> res
    ScoreAlignment(score=83.33333333333334, src_start=2, src_end=8, dest_start=0, dest_end=6)
    
    Using the alignment information it is possible to calculate the same fuzz.ratio

    >>> fuzz.ratio(s1[res.src_start:res.src_end], s2[res.dest_start:res.dest_end])
    83.33333333333334
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return None

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    res = partial_ratio_alignment_func(s1_proc.string, s2_proc.string, c_score_cutoff)

    if res.score >= c_score_cutoff:
        return ScoreAlignment(res.score, res.src_start, res.src_end, res.dest_start, res.dest_end)
    else:
        return None


def token_sort_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Sorts the words in the strings and calculates the fuzz.ratio between them

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.ratio

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
    (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.partial_ratio

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and
    fuzz.partial_token_sort_ratio (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def WRatio(s1, s2, *, processor=default_process, score_cutoff=None):
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return WRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def QRatio(s1, s2, *, processor=default_process, score_cutoff=None):
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
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is ``utils.default_process``.
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
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return QRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool GetScorerFlagsFuzzSymmetric(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsFuzz(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 100, "worst_score": 0, "flags": (1 << 5)}

cdef dict FuzzContextPy = CreateScorerContextPy(_GetScorerFlagsSimilarity)

cdef RF_Scorer RatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, RatioInit)
AddScorerContext(ratio, FuzzContextPy, &RatioContext)

cdef RF_Scorer PartialRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialRatioInit)
AddScorerContext(partial_ratio, FuzzContextPy, &PartialRatioContext)

cdef RF_Scorer TokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenSortRatioInit)
AddScorerContext(token_sort_ratio, FuzzContextPy, &TokenSortRatioContext)

cdef RF_Scorer TokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenSetRatioInit)
AddScorerContext(token_set_ratio, FuzzContextPy, &TokenSetRatioContext)

cdef RF_Scorer TokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenRatioInit)
AddScorerContext(token_ratio, FuzzContextPy, &TokenRatioContext)

cdef RF_Scorer PartialTokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSortRatioInit)
AddScorerContext(partial_token_sort_ratio, FuzzContextPy, &PartialTokenSortRatioContext)

cdef RF_Scorer PartialTokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSetRatioInit)
AddScorerContext(partial_token_set_ratio, FuzzContextPy, &PartialTokenSetRatioContext)

cdef RF_Scorer PartialTokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenRatioInit)
AddScorerContext(partial_token_ratio, FuzzContextPy, &PartialTokenRatioContext)

cdef RF_Scorer WRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, WRatioInit)
AddScorerContext(WRatio, FuzzContextPy, &WRatioContext)

cdef RF_Scorer QRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, QRatioInit)
AddScorerContext(QRatio, FuzzContextPy, &QRatioContext)
