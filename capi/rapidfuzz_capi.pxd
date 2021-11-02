from libcpp cimport bool

cdef extern from "rapidfuzz_capi.h":
    cdef enum RF_StringType:
        RF_UINT64
    
    ctypedef struct RF_String
    ctypedef void (*RF_StringDeinit) (RF_String* context)

    ctypedef struct RF_String:
        RF_StringType kind
        void* data
        size_t length
        void* context
        RF_StringDeinit deinit

    ctypedef struct RF_Kwargs
    ctypedef void (*RF_KwargsDeinit) (RF_Kwargs* context)

    ctypedef struct RF_Kwargs:
        void* context
        RF_KwargsDeinit deinit

    ctypedef bool (*RF_KwargsInit) (RF_Kwargs* context, dict kwargs) except False

    ctypedef struct RF_Similarity
    ctypedef bool (*RF_SimilarityFunc) (double* similarity, const RF_Similarity* context, const RF_String* str, double score_cutoff) nogil except False
    ctypedef void (*RF_SimilarityContextDeinit) (RF_Similarity* deinit) nogil

    ctypedef struct RF_Similarity:
        void* context
        RF_SimilarityFunc similarity
        RF_SimilarityContextDeinit deinit

    ctypedef bool (*RF_SimilarityInit) (RF_Similarity* context, const RF_Kwargs* kwargs, const RF_String* str) nogil except False

    ctypedef struct RfSimilarityFunctionTable:
        RF_KwargsInit kwargs_init
        RF_SimilarityInit similarity_init

    ctypedef struct RF_Distance
    ctypedef bool (*RF_DistanceFunc) (size_t* distance, const RF_Distance* context, const RF_String* str, size_t max)
    ctypedef void (*RF_DistanceContextDeinit) (RF_Distance* deinit) nogil

    ctypedef struct RF_Distance:
        void* context
        RF_DistanceFunc distance
        RF_DistanceContextDeinit deinit

    ctypedef bool (*RF_DistanceInit) (RF_Distance* context, const RF_Kwargs* kwargs, const RF_String* str) nogil except False

    ctypedef struct RfDistanceFunctionTable:
        RF_KwargsInit kwargs_init
        RF_DistanceInit distance_init
