# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

#from ._initialize_cpp import Editops
#from ._initialize_cpp cimport Editops, RfEditops

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
    double damerau_levenshtein_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t damerau_levenshtein_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double damerau_levenshtein_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t damerau_levenshtein_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    bool DamerauLevenshteinDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 1.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

cdef bool GetScorerFlagsDamerauLevenshteinDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsDamerauLevenshteinNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1
    return True

cdef bool GetScorerFlagsDamerauLevenshteinSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsDamerauLevenshteinNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer DamerauLevenshteinDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinDistance, DamerauLevenshteinDistanceInit)
distance._RF_Scorer = PyCapsule_New(&DamerauLevenshteinDistanceContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinNormalizedDistance, DamerauLevenshteinNormalizedDistanceInit)
normalized_distance._RF_Scorer = PyCapsule_New(&DamerauLevenshteinNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinSimilarity, DamerauLevenshteinSimilarityInit)
similarity._RF_Scorer = PyCapsule_New(&DamerauLevenshteinSimilarityContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinNormalizedSimilarity, DamerauLevenshteinNormalizedSimilarityInit)
normalized_similarity._RF_Scorer = PyCapsule_New(&DamerauLevenshteinNormalizedSimilarityContext, NULL, NULL)
