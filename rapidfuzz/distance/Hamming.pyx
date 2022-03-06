# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence
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

cdef inline void preprocess_strings(s1, s2, processor, RF_StringWrapper* s1_proc, RF_StringWrapper* s2_proc) except *:
    cdef RF_Preprocessor* preprocess_context = NULL

    if processor is None:
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
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

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return hamming_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

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

cdef RF_Scorer HammingDistanceContext
HammingDistanceContext.version = SCORER_STRUCT_VERSION
HammingDistanceContext.kwargs_init = NoKwargsInit
HammingDistanceContext.get_scorer_flags = GetScorerFlagsHammingDistance
HammingDistanceContext.scorer_func_init = HammingDistanceInit
distance._RF_Scorer = PyCapsule_New(&HammingDistanceContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedDistanceContext
HammingNormalizedDistanceContext.version = SCORER_STRUCT_VERSION
HammingNormalizedDistanceContext.kwargs_init = NoKwargsInit
HammingNormalizedDistanceContext.get_scorer_flags = GetScorerFlagsHammingNormalizedDistance
HammingNormalizedDistanceContext.scorer_func_init = HammingNormalizedDistanceInit
normalized_distance._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer HammingSimilarityContext
HammingSimilarityContext.version = SCORER_STRUCT_VERSION
HammingSimilarityContext.kwargs_init = NoKwargsInit
HammingSimilarityContext.get_scorer_flags = GetScorerFlagsHammingSimilarity
HammingSimilarityContext.scorer_func_init = HammingSimilarityInit
similarity._RF_Scorer = PyCapsule_New(&HammingSimilarityContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedSimilarityContext
HammingNormalizedSimilarityContext.version = SCORER_STRUCT_VERSION
HammingNormalizedSimilarityContext.kwargs_init = NoKwargsInit
HammingNormalizedSimilarityContext.get_scorer_flags = GetScorerFlagsHammingNormalizedSimilarity
HammingNormalizedSimilarityContext.scorer_func_init = HammingNormalizedSimilarityInit
normalized_similarity._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)
