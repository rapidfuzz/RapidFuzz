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
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 1.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def editops(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = lcs_seq_editops_func(s1_proc.string, s2_proc.string)
    return ops

def opcodes(s1, s2, *, processor=None):
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
