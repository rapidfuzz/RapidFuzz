cdef extern from "rapidfuzz_capi.h":
    cdef enum RfStringType:
        RF_UINT64
    
    ctypedef struct RfString
    ctypedef void (*RF_StringDeinit) (RfString* context)

    ctypedef struct RfString:
        RfStringType kind
        void* data
        size_t length
        void* context
        RF_StringDeinit deinit

    ctypedef struct RfKwargsContext
    ctypedef void (*RF_KwargsContextDeinit) (RfKwargsContext* context)

    ctypedef struct RfKwargsContext:
        void* context
        RF_KwargsContextDeinit deinit

    ctypedef int (*RF_KwargsContextInit) (RfKwargsContext* context, dict kwargs) except -1

    ctypedef struct RfSimilarityContext
    ctypedef int (*RF_SimilarityFunc) (double* similarity, const RfSimilarityContext* context, const RfString* str, double score_cutoff) nogil except -1
    ctypedef void (*RF_SimilarityContextDeinit) (RfSimilarityContext* deinit) nogil

    ctypedef struct RfSimilarityContext:
        void* context
        RF_SimilarityFunc similarity
        RF_SimilarityContextDeinit deinit

    ctypedef int (*RF_SimilarityInit) (RfSimilarityContext* context, const RfKwargsContext* kwargs, const RfString* str) nogil except -1

    ctypedef struct RfSimilarityFunctionTable:
        RF_KwargsContextInit kwargs_init
        RF_SimilarityInit similarity_init

    ctypedef struct RfDistanceContext
    ctypedef int (*RF_DistanceFunc) (size_t* distance, const RfDistanceContext* context, const RfString* str, size_t max)
    ctypedef void (*RF_DistanceContextDeinit) (RfDistanceContext* deinit) nogil

    ctypedef struct RfDistanceContext:
        void* context
        RF_DistanceFunc distance
        RF_DistanceContextDeinit deinit

    ctypedef int (*RF_DistanceInit) (RfDistanceContext* context, const RfKwargsContext* kwargs, const RfString* str) nogil except -1

    ctypedef struct RfDistanceFunctionTable:
        RF_KwargsContextInit kwargs_init
        RF_DistanceInit distance_init
