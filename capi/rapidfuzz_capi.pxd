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

    ctypedef RfKwargsContext (*RF_KwargsContextInit) (object kwargs)

    ctypedef struct RfSimilarityContext
    ctypedef double (*RF_SimilarityFunc) (const RfSimilarityContext* context, const RfString* str, double score_cutoff)
    ctypedef void (*RF_SimilarityContextDeinit) (RfSimilarityContext* deinit)

    ctypedef struct RfSimilarityContext:
        void* context
        RF_SimilarityFunc similarity
        RF_SimilarityContextDeinit deinit

    ctypedef RfSimilarityContext (*RF_SimilarityInit) (const RfKwargsContext* kwargs, const RfString* str)

    ctypedef struct RfSimilarityFunctionTable:
        RF_KwargsContextInit kwargs_init
        RF_SimilarityInit similarity_init

    ctypedef struct RfDistanceContext
    ctypedef double (*RF_DistanceFunc) (const RfDistanceContext* context, const RfString* str, double score_cutoff)
    ctypedef void (*RF_DistanceContextDeinit) (RfDistanceContext* deinit)

    ctypedef struct RfDistanceContext:
        void* context
        RF_DistanceFunc distance
        RF_DistanceContextDeinit deinit

    ctypedef RfDistanceContext (*RF_DistanceInit) (const RfKwargsContext* kwargs, const RfString* str)

    ctypedef struct RfDistanceFunctionTable:
        RF_KwargsContextInit kwargs_init
        RF_DistanceInit distance_init
