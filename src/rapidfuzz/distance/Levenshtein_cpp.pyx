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
from cpp_common cimport RF_StringWrapper, preprocess_strings, CreateScorerContext

from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdint cimport INT64_MAX, int64_t
from cpython.pycapsule cimport PyCapsule_New
from cython.operator cimport dereference

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cdef struct LevenshteinWeightTable:
        int64_t insert_cost
        int64_t delete_cost
        int64_t replace_cost

cdef extern from "edit_based.hpp":
    double levenshtein_normalized_distance_func(  const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_distance_func(            const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +
    double levenshtein_normalized_similarity_func(const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_similarity_func(          const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +

    RfEditops levenshtein_editops_func(const RF_String&, const RF_String&, int64_t) nogil except +

    bool LevenshteinDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

cdef int64_t get_score_cutoff_i64(score_cutoff, int64_t default) except -1:
    cdef int64_t c_score_cutoff = default
    if score_cutoff is None:
        return c_score_cutoff

    c_score_cutoff = score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    return c_score_cutoff

cdef double get_score_cutoff_f64(score_cutoff, double default) except -1:
    cdef double c_score_cutoff = default
    if score_cutoff is None:
        return c_score_cutoff

    c_score_cutoff = score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    return c_score_cutoff

cdef int64_t get_score_hint_i64(score_hint, int64_t default) except -1:
    cdef int64_t c_score_hint = default
    if score_hint is None:
        return c_score_hint

    c_score_hint = score_hint
    if c_score_hint < 0:
        raise ValueError("score_hint has to be >= 0")

    return c_score_hint


def distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_distance_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_similarity_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def normalized_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_normalized_distance_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def normalized_similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_normalized_similarity_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def editops(s1, s2, *, processor=None, score_hint=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)
    cdef int64_t c_score_hint = get_score_hint_i64(score_hint, INT64_MAX)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string, c_score_hint)
    return ops


def opcodes(s1, s2, *, processor=None, score_hint=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)
    cdef int64_t c_score_hint = get_score_hint_i64(score_hint, INT64_MAX)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string, c_score_hint)
    return ops.as_opcodes()

cdef void KwargsDeinit(RF_Kwargs* self):
    free(<void*>dereference(self).context)

cdef bool LevenshteinKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    cdef int64_t insertion, deletion, substitution
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>malloc(sizeof(LevenshteinWeightTable))

    if not weights:
        raise MemoryError

    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    dereference(weights).insert_cost = insertion
    dereference(weights).delete_cost = deletion
    dereference(weights).replace_cost = substitution
    dereference(self).context = weights
    dereference(self).dtor = KwargsDeinit
    return True

cdef bool GetScorerFlagsDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0
    dereference(scorer_flags).worst_score.f64 = 1.0
    return True

cdef bool GetScorerFlagsNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LevenshteinDistanceContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsDistance, LevenshteinDistanceInit)
distance._RF_Scorer = PyCapsule_New(&LevenshteinDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinSimilarityContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsSimilarity, LevenshteinSimilarityInit)
similarity._RF_Scorer = PyCapsule_New(&LevenshteinSimilarityContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedDistanceContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsNormalizedDistance, LevenshteinNormalizedDistanceInit)
normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedSimilarityContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsNormalizedSimilarity, LevenshteinNormalizedSimilarityInit)
normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedSimilarityContext, NULL, NULL)
