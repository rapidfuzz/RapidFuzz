from libcpp cimport bool

cdef extern from "rapidfuzz_capi.h":
    cdef enum RF_StringType:
        RF_UINT64

    ctypedef struct RF_String:
        void (*dtor) (RF_String* self)

        RF_StringType kind
        void* data
        size_t length
        void* context

    ctypedef struct RF_Kwargs:
        void (*dtor) (RF_Kwargs* self)

        void* context

    ctypedef bool (*RF_KwargsInit) (RF_Kwargs* context, dict kwargs) except False

    ctypedef struct RF_Similarity:
        void (*dtor) (RF_Similarity* self)
        bool (*similarity) (const RF_Similarity* context, const RF_String* str, double score_cutoff, double* similarity) nogil except False

        void* context

    ctypedef bool (*RF_SimilarityInit) (RF_Similarity* context, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings) nogil except False

    ctypedef struct RfSimilarityFunctionTable:
        RF_KwargsInit kwargs_init
        RF_SimilarityInit similarity_init

    ctypedef struct RF_Distance:
        void (*dtor) (RF_Distance* self)
        bool (*distance) (const RF_Distance* self, const RF_String* str, size_t max, size_t* distance) nogil except False

        void* context

    ctypedef bool (*RF_DistanceInit) (RF_Distance* context, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings) nogil except False

    ctypedef struct RfDistanceFunctionTable:
        RF_KwargsInit kwargs_init
        RF_DistanceInit distance_init
