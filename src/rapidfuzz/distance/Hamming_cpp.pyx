# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_SYMMETRIC
)

# required for preprocess_strings
from array import array
from cpp_common cimport RF_StringWrapper, preprocess_strings, NoKwargsInit, CreateScorerContext
from libc.stdint cimport INT64_MAX, int64_t

from libcpp cimport bool
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "edit_based.hpp":
    double hamming_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t hamming_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double hamming_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t hamming_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    bool HammingDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def distance(s1, s2, *, processor=None, score_cutoff=None):
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
    score_cutoff : int or None, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If s1 and s2 have a different length
    """
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the Hamming similarity between two strings.
    
    This is calculated as ``len1 - distance``.

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
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If s1 and s2 have a different length
    """
    cdef int64_t c_score_cutoff = 0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein similarity in the range [1, 0].

    This is calculated as ``distance / (len1 + len2)``.

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
    return hamming_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized hamming similarity in the range [0, 1].

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
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool GetScorerFlagsHammingDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsHammingNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1.0
    return True

cdef bool GetScorerFlagsHammingSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsHammingNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer HammingDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingDistance, HammingDistanceInit)
distance._RF_Scorer = PyCapsule_New(&HammingDistanceContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingNormalizedDistance, HammingNormalizedDistanceInit)
normalized_distance._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer HammingSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingSimilarity, HammingSimilarityInit)
similarity._RF_Scorer = PyCapsule_New(&HammingSimilarityContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingNormalizedSimilarity, HammingNormalizedSimilarityInit)
normalized_similarity._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)

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
