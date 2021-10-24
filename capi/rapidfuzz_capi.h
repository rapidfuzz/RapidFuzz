#ifndef RAPIDFUZZ_CAPI_H
#define RAPIDFUZZ_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"

#include <stddef.h>

#define RF_PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

enum RfStringType {
#if PY_VERSION_HEX > RF_PYTHON_VERSION(3, 0, 0)
    RF_UINT8,  /* uint8_t */
    RF_UINT16, /* uint16_t */
    RF_UINT32, /* uint32_t */
    RF_UINT64  /* uint64_t */
#else /* Python2 */
    RF_CHAR,    /* char */
    RF_UNICODE, /* Py_UNICODE */
    RF_UINT64   /* uint64_t */
#endif
};

typedef void (*RF_StringDeinit) (struct _RfString* context);

typedef struct _RfString {
    RfStringType kind;
    void* data;
    size_t length;
    void* context;
    RF_StringDeinit deinit;
} RfString;

typedef void (*RF_KwargsContextDeinit) (struct _RfKwargsContext* context);

typedef struct _RfKwargsContext {
    void* context;
    RF_KwargsContextDeinit deinit;
} RfKwargsContext;

typedef RfKwargsContext (*RF_KwargsContextInit)(PyObject* kwargs);

typedef double (*RF_SimilarityFunc) (const struct _RfSimilarityContext* context, const RfString* str, double score_cutoff);
typedef void (*RF_SimilarityContextDeinit) (struct _RfSimilarityContext* context);

typedef struct _RfSimilarityContext {
    void* context;
    RF_SimilarityFunc similarity;
    RF_SimilarityContextDeinit deinit;
} RfSimilarityContext;

typedef RfSimilarityContext (*RF_SimilarityInit) (const RfKwargsContext* kwargs, const RfString* str);

typedef struct {
    RF_KwargsContextInit kwargs_init;
    RF_SimilarityInit similarity_init;
} RfSimilarityFunctionTable;

typedef double (*RF_DistanceFunc) (const struct _RfDistanceContext* context, const RfString* str, double score_cutoff);
typedef void (*RF_DistanceContextDeinit) (struct _RfDistanceContext* context);

typedef struct _RfDistanceContext {
    void* context;
    RF_DistanceFunc distance;
    RF_DistanceContextDeinit deinit;
} RfDistanceContext;

typedef RfDistanceContext (*RF_DistanceInit) (const RfKwargsContext* kwargs, const RfString* str);

typedef struct {
    RF_KwargsContextInit kwargs_init;
    RF_DistanceInit distance_init;
} RfDistanceFunctionTable;

#ifdef __cplusplus
}
#endif

#endif  /* RAPIDFUZZ_CAPI_H */
