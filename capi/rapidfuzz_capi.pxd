from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t

cdef extern from "rapidfuzz_capi.h":
    cdef enum RF_StringType:
        RF_UINT64

    uint32_t RF_SCORER_FLAG_MULTI_STRING
    uint32_t RF_SCORER_FLAG_RESULT_F64
    uint32_t RF_SCORER_FLAG_RESULT_I64
    uint32_t RF_SCORER_FLAG_RESULT_U64
    uint32_t RF_SCORER_FLAG_SYMMETRIC
    uint32_t RF_SCORER_FLAG_TRIANGLE_INEQUALITY

    ctypedef struct RF_String:
        void (*dtor) (RF_String*) nogil

        RF_StringType kind
        void* data
        size_t length
        void* context

    ctypedef bool (*RF_Preprocess) (object, RF_String*) except False

    uint32_t PREPROCESSOR_STRUCT_VERSION

    ctypedef struct RF_Preprocessor:
        uint32_t version
        RF_Preprocess preprocess

    ctypedef struct RF_Kwargs:
        void (*dtor) (RF_Kwargs*)

        void* context

    ctypedef bool (*RF_KwargsInit) (RF_Kwargs*, dict) except False

    ctypedef union _RF_ScorerFunc_union:
        bool (*f64) (const RF_ScorerFunc*, const RF_String*, double, double*) nogil except False
        bool (*u64) (const RF_ScorerFunc*, const RF_String*, uint64_t, uint64_t*) nogil except False
        bool (*i64) (const RF_ScorerFunc*, const RF_String*, int64_t, int64_t*) nogil except False

    ctypedef struct RF_ScorerFunc:
        void (*dtor) (RF_ScorerFunc*) nogil
        _RF_ScorerFunc_union call

        void* context

    ctypedef bool (*RF_ScorerFuncInit) (RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False

    ctypedef union _RF_RF_ScorerFlags_OptimalScore_union:
        double   f64
        uint64_t u64
        int64_t  i64

    ctypedef union _RF_RF_ScorerFlags_WorstScore_union:
        double   f64
        uint64_t u64
        int64_t  i64

    ctypedef struct RF_ScorerFlags:
        uint32_t flags
        _RF_RF_ScorerFlags_OptimalScore_union optimal_score
        _RF_RF_ScorerFlags_WorstScore_union worst_score

    ctypedef bool (*RF_GetScorerFlags) (const RF_Kwargs*, RF_ScorerFlags*) nogil except False

    uint32_t SCORER_STRUCT_VERSION

    ctypedef struct RF_Scorer:
        uint32_t version
        RF_KwargsInit kwargs_init
        RF_GetScorerFlags get_scorer_flags
        RF_ScorerFuncInit scorer_func_init
