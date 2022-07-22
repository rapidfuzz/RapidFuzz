# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from ._initialize_cpp import Editops
from ._initialize_cpp cimport Editops, RfEditops

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_SYMMETRIC
)
# required for preprocess_strings
from array import array
from cpp_common cimport RF_StringWrapper, preprocess_strings, NoKwargsInit, CreateScorerContext

from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdint cimport INT64_MAX, uint32_t, int64_t
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "edit_based.hpp":
    double lcs_seq_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t lcs_seq_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double lcs_seq_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t lcs_seq_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    RfEditops lcs_seq_editops_func(const RF_String&, const RF_String&) nogil except +

    bool LCSseqDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def distance(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the LCS distance in the range [0, max].

    This is calculated as ``max(len1, len2) - similarity``.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Examples
    --------
    Find the LCS distance between two strings:

    >>> from rapidfuzz.distance import LCSseq
    >>> LCSseq.distance("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> LCSseq.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the length of the longest common subsequence

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the similarity is smaller than score_cutoff,
        0 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    similarity : int
        similarity between s1 and s2
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized LCS similarity in the range [1, 0].

    This is calculated as ``distance / max(len1, len2)``.

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
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_dist > score_cutoff 1.0 is returned instead. Default is 1.0,
        which deactivates this behaviour.

    Returns
    -------
    norm_dist : float
        normalized distance between s1 and s2 as a float between 0 and 1.0
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 1.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized LCS similarity in the range [0, 1].

    This is calculated as ``1 - normalized_distance``

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
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_sim < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    norm_sim : float
        normalized similarity between s1 and s2 as a float between 0 and 1.0

    Examples
    --------
    Find the normalized LCS similarity between two strings:

    >>> from rapidfuzz.distance import LCSseq
    >>> LCSseq.normalized_similarity("lewenstein", "levenshtein")
    0.8181818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> LCSseq.normalized_similarity("lewenstein", "levenshtein", score_cutoff=0.9)
    0.0

    When a different processor is used s1 and s2 do not have to be strings

    >>> LCSseq.normalized_similarity(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def editops(s1, s2, *, processor=None):
    """
    Return Editops describing how to turn s1 into s2.

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
    editops : Editops
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [6]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [6] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import LCSseq
    >>> for tag, src_pos, dest_pos in LCSseq.editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[0] s2[0]
     delete s1[3] s2[2]
     insert s1[4] s2[2]
     insert s1[6] s2[5]
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = lcs_seq_editops_func(s1_proc.string, s2_proc.string)
    return ops

def opcodes(s1, s2, *, processor=None):
    """
    Return Opcodes describing how to turn s1 into s2.

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
    opcodes : Opcodes
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [7]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [7] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import LCSseq

    >>> a = "qabxcd"
    >>> b = "abycdf"
    >>> for tag, i1, i2, j1, j2 in LCSseq.opcodes(a, b):
    ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
    ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
     delete a[0:1] (q) b[0:0] ()
      equal a[1:3] (ab) b[0:2] (ab)
     delete a[3:4] (x) b[2:2] ()
     insert a[4:4] () b[2:3] (y)
      equal a[4:6] (cd) b[3:5] (cd)
     insert a[6:6] () b[5:6] (f)
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = lcs_seq_editops_func(s1_proc.string, s2_proc.string)
    return ops.as_opcodes()


cdef bool GetScorerFlagsLCSseqDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsLCSseqNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1
    return True

cdef bool GetScorerFlagsLCSseqSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsLCSseqNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LCSseqDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqDistance, LCSseqDistanceInit)
distance._RF_Scorer = PyCapsule_New(&LCSseqDistanceContext, NULL, NULL)

cdef RF_Scorer LCSseqNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqNormalizedDistance, LCSseqNormalizedDistanceInit)
normalized_distance._RF_Scorer = PyCapsule_New(&LCSseqNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer LCSseqSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqSimilarity, LCSseqSimilarityInit)
similarity._RF_Scorer = PyCapsule_New(&LCSseqSimilarityContext, NULL, NULL)

cdef RF_Scorer LCSseqNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqNormalizedSimilarity, LCSseqNormalizedSimilarityInit)
normalized_similarity._RF_Scorer = PyCapsule_New(&LCSseqNormalizedSimilarityContext, NULL, NULL)

def _GetScorerFlagsDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 2**63 - 1, "flags": (1 << 6)}


def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 2**63 - 1, "worst_score": 0, "flags": (1 << 6)}


def _GetScorerFlagsNormalizedDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 1, "flags": (1 << 5)}


def _GetScorerFlagsNormalizedSimilarity(**kwargs):
    return {"optimal_score": 1, "worst_score": 0, "flags": (1 << 5)}


distance._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsDistance}

similarity._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

normalized_distance._RF_ScorerPy = {
    "get_scorer_flags": _GetScorerFlagsNormalizedDistance
}

normalized_similarity._RF_ScorerPy = {
    "get_scorer_flags": _GetScorerFlagsNormalizedSimilarity
}
