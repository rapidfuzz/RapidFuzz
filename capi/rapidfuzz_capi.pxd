from libcpp cimport bool

cdef extern from "rapidfuzz_capi.h":
    cdef enum RF_StringType:
        RF_UINT64

    cdef enum RF_ScorerType:
        RF_DISTANCE
        RF_SIMILARITY

    ctypedef struct RF_String:
        void (*dtor) (RF_String* self) nogil

        RF_StringType kind
        void* data
        size_t length
        void* context

    ctypedef bool (*RF_Preprocess) (object obj, RF_String* str) except False

    ctypedef struct RF_Kwargs:
        void (*dtor) (RF_Kwargs* self)

        void* context

    ctypedef bool (*RF_KwargsInit) (RF_Kwargs* self, dict kwargs) except False

    ctypedef struct RF_Similarity:
        void (*dtor) (RF_Similarity* self) nogil
        bool (*similarity) (const RF_Similarity* self, const RF_String* str, double score_cutoff, double* similarity) nogil except False

        void* context

    ctypedef bool (*RF_SimilarityInit) (RF_Similarity* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings) nogil except False

    ctypedef struct RF_Distance:
        void (*dtor) (RF_Distance* self) nogil
        bool (*distance) (const RF_Distance* self, const RF_String* str, size_t max, size_t* distance) nogil except False

        void* context

    ctypedef bool (*RF_DistanceInit) (RF_Distance* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings) nogil except False

    ctypedef union _RF_Scorer_union:
        RF_DistanceInit distance_init
        RF_SimilarityInit similarity_init

    ctypedef struct RF_Scorer:
        RF_ScorerType scorer_type
        RF_KwargsInit kwargs_init

        _RF_Scorer_union scorer
