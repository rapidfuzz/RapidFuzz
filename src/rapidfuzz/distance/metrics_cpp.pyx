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
from libc.stdint cimport INT64_MAX, int64_t
from cpython.pycapsule cimport PyCapsule_New
from cython.operator cimport dereference

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cdef struct LevenshteinWeightTable:
        int64_t insert_cost
        int64_t delete_cost
        int64_t replace_cost

cdef extern from "metrics.hpp":
    # Levenshtein
    double levenshtein_normalized_distance_func(  const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_distance_func(            const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +
    double levenshtein_normalized_similarity_func(const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_similarity_func(          const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +

    RfEditops levenshtein_editops_func(const RF_String&, const RF_String&, int64_t) nogil except +

    bool LevenshteinDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

    # Damerau Levenshtein
    double damerau_levenshtein_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t damerau_levenshtein_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double damerau_levenshtein_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t damerau_levenshtein_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    bool DamerauLevenshteinDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool DamerauLevenshteinNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

    # LCS
    double lcs_seq_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t lcs_seq_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double lcs_seq_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t lcs_seq_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    RfEditops lcs_seq_editops_func(const RF_String&, const RF_String&) nogil except +

    bool LCSseqDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LCSseqNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

    # Indel
    double indel_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t indel_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double indel_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t indel_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    RfEditops indel_editops_func(const RF_String&, const RF_String&) nogil except +

    bool IndelDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

    # Hamming
    double hamming_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t hamming_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double hamming_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t hamming_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    bool HammingDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

    RfEditops hamming_editops_func(const RF_String&, const RF_String&) nogil except +

    # Damerau Levenshtein
    double osa_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t osa_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double osa_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t osa_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    bool OSADistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool OSANormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool OSASimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool OSANormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False


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


def levenshtein_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_distance_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def levenshtein_similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return levenshtein_similarity_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def levenshtein_normalized_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
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


def levenshtein_normalized_similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
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


def levenshtein_editops(s1, s2, *, processor=None, score_hint=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)
    cdef int64_t c_score_hint = get_score_hint_i64(score_hint, INT64_MAX)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string, c_score_hint)
    return ops


def levenshtein_opcodes(s1, s2, *, processor=None, score_hint=None):
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

cdef bool GetScorerFlagsLevenshteinDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsLevenshteinSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsLevenshteinNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0
    dereference(scorer_flags).worst_score.f64 = 1.0
    return True

cdef bool GetScorerFlagsLevenshteinNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LevenshteinDistanceContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsLevenshteinDistance, LevenshteinDistanceInit)
levenshtein_distance._RF_Scorer = PyCapsule_New(&LevenshteinDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinSimilarityContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsLevenshteinSimilarity, LevenshteinSimilarityInit)
levenshtein_similarity._RF_Scorer = PyCapsule_New(&LevenshteinSimilarityContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedDistanceContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsLevenshteinNormalizedDistance, LevenshteinNormalizedDistanceInit)
levenshtein_normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedSimilarityContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsLevenshteinNormalizedSimilarity, LevenshteinNormalizedSimilarityInit)
levenshtein_normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedSimilarityContext, NULL, NULL)


def damerau_levenshtein_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def damerau_levenshtein_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def damerau_levenshtein_normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return damerau_levenshtein_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def damerau_levenshtein_normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
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
damerau_levenshtein_distance._RF_Scorer = PyCapsule_New(&DamerauLevenshteinDistanceContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinNormalizedDistance, DamerauLevenshteinNormalizedDistanceInit)
damerau_levenshtein_normalized_distance._RF_Scorer = PyCapsule_New(&DamerauLevenshteinNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinSimilarity, DamerauLevenshteinSimilarityInit)
damerau_levenshtein_similarity._RF_Scorer = PyCapsule_New(&DamerauLevenshteinSimilarityContext, NULL, NULL)

cdef RF_Scorer DamerauLevenshteinNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsDamerauLevenshteinNormalizedSimilarity, DamerauLevenshteinNormalizedSimilarityInit)
damerau_levenshtein_normalized_similarity._RF_Scorer = PyCapsule_New(&DamerauLevenshteinNormalizedSimilarityContext, NULL, NULL)


def lcs_seq_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def lcs_seq_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def lcs_seq_normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def lcs_seq_normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return lcs_seq_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def lcs_seq_editops(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = lcs_seq_editops_func(s1_proc.string, s2_proc.string)
    return ops


def lcs_seq_opcodes(s1, s2, *, processor=None):
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
lcs_seq_distance._RF_Scorer = PyCapsule_New(&LCSseqDistanceContext, NULL, NULL)

cdef RF_Scorer LCSseqNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqNormalizedDistance, LCSseqNormalizedDistanceInit)
lcs_seq_normalized_distance._RF_Scorer = PyCapsule_New(&LCSseqNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer LCSseqSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqSimilarity, LCSseqSimilarityInit)
lcs_seq_similarity._RF_Scorer = PyCapsule_New(&LCSseqSimilarityContext, NULL, NULL)

cdef RF_Scorer LCSseqNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsLCSseqNormalizedSimilarity, LCSseqNormalizedSimilarityInit)
lcs_seq_normalized_similarity._RF_Scorer = PyCapsule_New(&LCSseqNormalizedSimilarityContext, NULL, NULL)


def indel_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return indel_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def indel_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    cdef RF_StringWrapper s1_proc, s2_proc
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return indel_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def indel_normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return indel_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def indel_normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return indel_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def indel_editops(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = indel_editops_func(s1_proc.string, s2_proc.string)
    return ops


def indel_opcodes(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = indel_editops_func(s1_proc.string, s2_proc.string)
    return ops.as_opcodes()


cdef bool GetScorerFlagsIndelDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True


cdef bool GetScorerFlagsIndelNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1
    return True

cdef bool GetScorerFlagsIndelSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True


cdef bool GetScorerFlagsIndelNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True


cdef RF_Scorer IndelDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsIndelDistance, IndelDistanceInit)
indel_distance._RF_Scorer = PyCapsule_New(&IndelDistanceContext, NULL, NULL)

cdef RF_Scorer IndelNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsIndelNormalizedDistance, IndelNormalizedDistanceInit)
indel_normalized_distance._RF_Scorer = PyCapsule_New(&IndelNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer IndelSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsIndelSimilarity, IndelSimilarityInit)
indel_similarity._RF_Scorer = PyCapsule_New(&IndelSimilarityContext, NULL, NULL)

cdef RF_Scorer IndelNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsIndelNormalizedSimilarity, IndelNormalizedSimilarityInit)
indel_normalized_similarity._RF_Scorer = PyCapsule_New(&IndelNormalizedSimilarityContext, NULL, NULL)


def hamming_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def hamming_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def hamming_normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def hamming_normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return hamming_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def hamming_editops(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = hamming_editops_func(s1_proc.string, s2_proc.string)
    return ops


def hamming_opcodes(s1, s2, *, processor=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    ops.editops = hamming_editops_func(s1_proc.string, s2_proc.string)
    return ops.as_opcodes()


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
hamming_distance._RF_Scorer = PyCapsule_New(&HammingDistanceContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingNormalizedDistance, HammingNormalizedDistanceInit)
hamming_normalized_distance._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer HammingSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingSimilarity, HammingSimilarityInit)
hamming_similarity._RF_Scorer = PyCapsule_New(&HammingSimilarityContext, NULL, NULL)

cdef RF_Scorer HammingNormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHammingNormalizedSimilarity, HammingNormalizedSimilarityInit)
hamming_normalized_similarity._RF_Scorer = PyCapsule_New(&HammingNormalizedDistanceContext, NULL, NULL)

def osa_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, INT64_MAX)
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return osa_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def osa_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef int64_t c_score_cutoff = get_score_cutoff_i64(score_cutoff, 0)
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return osa_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def osa_normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 1.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return osa_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def osa_normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = get_score_cutoff_f64(score_cutoff, 0.0)
    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, None)
    return osa_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool GetScorerFlagsOSADistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsOSANormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1.0
    return True

cdef bool GetScorerFlagsOSASimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsOSANormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer OSADistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsOSADistance, OSADistanceInit)
osa_distance._RF_Scorer = PyCapsule_New(&OSADistanceContext, NULL, NULL)

cdef RF_Scorer OSANormalizedDistanceContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsOSANormalizedDistance, OSANormalizedDistanceInit)
osa_normalized_distance._RF_Scorer = PyCapsule_New(&OSANormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer OSASimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsOSASimilarity, OSASimilarityInit)
osa_similarity._RF_Scorer = PyCapsule_New(&OSASimilarityContext, NULL, NULL)

cdef RF_Scorer OSANormalizedSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsOSANormalizedSimilarity, OSANormalizedSimilarityInit)
osa_normalized_similarity._RF_Scorer = PyCapsule_New(&OSANormalizedDistanceContext, NULL, NULL)
