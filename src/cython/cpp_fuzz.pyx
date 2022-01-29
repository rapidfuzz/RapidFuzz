# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array
from rapidfuzz.utils import default_process

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_ScorerFunc, RF_Kwargs,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_SYMMETRIC
)

from cpp_common cimport (
    RF_StringWrapper, is_valid_string, convert_string, hash_array, hash_sequence,
    conv_sequence
)

from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cython.operator cimport dereference

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer

from array import array

cdef extern from "cpp_fuzz.hpp":
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

cdef inline void preprocess_strings(s1, s2, processor, RF_StringWrapper* s1_proc, RF_StringWrapper* s2_proc) except *:
    cdef RF_Preprocessor* preprocess_context = NULL

    if processor is True:
        # todo: deprecate
        processor = default_process

    if not processor:
        s1_proc[0] = RF_StringWrapper(conv_sequence(s1))
        s2_proc[0] = RF_StringWrapper(conv_sequence(s2))
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            preprocess_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        if preprocess_context != NULL and preprocess_context.version == 1:
            preprocess_context.preprocess(s1, &(s1_proc[0].string))
            preprocess_context.preprocess(s2, &(s2_proc[0].string))
        else:
            s1 = processor(s1)
            s1_proc[0] = RF_StringWrapper(conv_sequence(s1), s1)
            s2 = processor(s2)
            s2_proc[0] = RF_StringWrapper(conv_sequence(s2), s2)

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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return partial_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return QRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)

cdef bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

cdef bool GetScorerFlagsRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsPartialRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsTokenSortRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsTokenSetRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsTokenRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsPartialTokenSortRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsPartialTokenSetRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsPartialTokenRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsWRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsQRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer RatioContext
RatioContext.version = SCORER_STRUCT_VERSION
RatioContext.kwargs_init = NoKwargsInit
RatioContext.get_scorer_flags = GetScorerFlagsRatio
RatioContext.scorer_func_init = RatioInit
ratio._RF_Scorer = PyCapsule_New(&RatioContext, NULL, NULL)

cdef RF_Scorer PartialRatioContext
PartialRatioContext.version = SCORER_STRUCT_VERSION
PartialRatioContext.kwargs_init = NoKwargsInit
PartialRatioContext.get_scorer_flags = GetScorerFlagsPartialRatio
PartialRatioContext.scorer_func_init = PartialRatioInit
partial_ratio._RF_Scorer = PyCapsule_New(&PartialRatioContext, NULL, NULL)

cdef RF_Scorer TokenSortRatioContext
TokenSortRatioContext.version = SCORER_STRUCT_VERSION
TokenSortRatioContext.kwargs_init = NoKwargsInit
TokenSortRatioContext.get_scorer_flags = GetScorerFlagsTokenSortRatio
TokenSortRatioContext.scorer_func_init = TokenSortRatioInit
token_sort_ratio._RF_Scorer = PyCapsule_New(&TokenSortRatioContext, NULL, NULL)

cdef RF_Scorer TokenSetRatioContext
TokenSetRatioContext.version = SCORER_STRUCT_VERSION
TokenSetRatioContext.kwargs_init = NoKwargsInit
TokenSetRatioContext.get_scorer_flags = GetScorerFlagsTokenSetRatio
TokenSetRatioContext.scorer_func_init = TokenSetRatioInit
token_set_ratio._RF_Scorer = PyCapsule_New(&TokenSetRatioContext, NULL, NULL)

cdef RF_Scorer TokenRatioContext
TokenRatioContext.version = SCORER_STRUCT_VERSION
TokenRatioContext.kwargs_init = NoKwargsInit
TokenRatioContext.get_scorer_flags = GetScorerFlagsTokenRatio
TokenRatioContext.scorer_func_init = TokenRatioInit
token_ratio._RF_Scorer = PyCapsule_New(&TokenRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenSortRatioContext
PartialTokenSortRatioContext.version = SCORER_STRUCT_VERSION
PartialTokenSortRatioContext.kwargs_init = NoKwargsInit
PartialTokenSortRatioContext.get_scorer_flags = GetScorerFlagsPartialTokenSortRatio
PartialTokenSortRatioContext.scorer_func_init = PartialTokenSortRatioInit
partial_token_sort_ratio._RF_Scorer = PyCapsule_New(&PartialTokenSortRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenSetRatioContext
PartialTokenSetRatioContext.version = SCORER_STRUCT_VERSION
PartialTokenSetRatioContext.kwargs_init = NoKwargsInit
PartialTokenSetRatioContext.get_scorer_flags = GetScorerFlagsPartialTokenSetRatio
PartialTokenSetRatioContext.scorer_func_init = PartialTokenSetRatioInit
partial_token_set_ratio._RF_Scorer = PyCapsule_New(&PartialTokenSetRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenRatioContext
PartialTokenRatioContext.version = SCORER_STRUCT_VERSION
PartialTokenRatioContext.kwargs_init = NoKwargsInit
PartialTokenRatioContext.get_scorer_flags = GetScorerFlagsPartialTokenRatio
PartialTokenRatioContext.scorer_func_init = PartialTokenRatioInit
partial_token_ratio._RF_Scorer = PyCapsule_New(&PartialTokenRatioContext, NULL, NULL)

cdef RF_Scorer WRatioContext
WRatioContext.version = SCORER_STRUCT_VERSION
WRatioContext.kwargs_init = NoKwargsInit
WRatioContext.get_scorer_flags = GetScorerFlagsWRatio
WRatioContext.scorer_func_init = WRatioInit
WRatio._RF_Scorer = PyCapsule_New(&WRatioContext, NULL, NULL)

cdef RF_Scorer QRatioContext
QRatioContext.version = SCORER_STRUCT_VERSION
QRatioContext.kwargs_init = NoKwargsInit
QRatioContext.get_scorer_flags = GetScorerFlagsQRatio
QRatioContext.scorer_func_init = QRatioInit
QRatio._RF_Scorer = PyCapsule_New(&QRatioContext, NULL, NULL)
