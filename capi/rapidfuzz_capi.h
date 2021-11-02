#ifndef RAPIDFUZZ_CAPI_H
#define RAPIDFUZZ_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"

#include <stddef.h>
#include <stdbool.h>

#define RF_PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

enum RF_StringType {
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

typedef void (*RF_StringDeinit) (struct _RF_String* context);

typedef struct _RF_String {
    RF_StringType kind;
    void* data;
    size_t length;
    void* context;
    RF_StringDeinit deinit;
} RF_String;

typedef void (*RF_KwargsDeinit) (struct _RF_Kwargs* context);

typedef struct _RF_Kwargs {
    void* context;
    RF_KwargsDeinit deinit;
} RF_Kwargs;

typedef bool (*RF_KwargsInit)(RF_Kwargs* context, PyObject* kwargs);

typedef bool (*RF_SimilarityFunc) (double* similarity, const struct _RF_Similarity* context, const RF_String* str, double score_cutoff);
typedef void (*RF_SimilarityContextDeinit) (struct _RF_Similarity* context);

typedef struct _RF_Similarity {
    void* context;
    RF_SimilarityFunc similarity;
    RF_SimilarityContextDeinit deinit;
} RF_Similarity;

typedef bool (*RF_SimilarityInit) (RF_Similarity* context, const RF_Kwargs* kwargs, const RF_String* str);

typedef struct {
    RF_KwargsInit kwargs_init;
    RF_SimilarityInit similarity_init;
} RfSimilarityFunctionTable;

typedef bool (*RF_DistanceFunc) (size_t* distance, const struct _RF_Distance* context, const RF_String* str, size_t max);
typedef void (*RF_DistanceContextDeinit) (struct _RF_Distance* context);

typedef struct _RF_Distance {
    void* context;
    RF_DistanceFunc distance;
    RF_DistanceContextDeinit deinit;
} RF_Distance;

typedef bool (*RF_DistanceInit) (RF_Distance* context, const RF_Kwargs* kwargs, const RF_String* str);

typedef struct {
    RF_KwargsInit kwargs_init;
    RF_DistanceInit distance_init;
} RfDistanceFunctionTable;

#ifdef __cplusplus
}
#endif

#endif  /* RAPIDFUZZ_CAPI_H */
