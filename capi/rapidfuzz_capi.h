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
    /**
     * @brief destructor for RF_String
     * 
     * @param self pointer to RF_String instance to destruct
     */
    void (*dtor) (struct _RF_String* self);

/* members */
    RF_StringType kind;
    void* data;
    size_t length;
    void* context;
} RF_String;

typedef struct _RF_Kwargs {
    void (*dtor) (struct _RF_Kwargs* self);

/* members */
    void* context;
} RF_Kwargs;

typedef bool (*RF_KwargsInit)(RF_Kwargs* context, PyObject* kwargs);

typedef struct _RF_Similarity {
    /**
     * @brief destructor for RF_Similarity object
     * 
     * @param self pointer to RF_Similarity instance to destruct
     */
    void (*dtor) (struct _RF_Similarity* self);

    /**
     * @brief Calculate similarity between 0 and 100
     * 
     * @param[in] self pointer to RF_Distance instance
     * @param[in] str string to calculate similarity with strings passed into `ctor`
     * @param[in] score_cutoff argument for a score threshold between 0 and 100.
     * When the similarity is < score_cutoff the similarity is 0.
     * @param[out] similarity array of size `str_count` for results of the similarity calculation
     * 
     * @return true on success and false with a Python exception set on failure
     */
    bool (*similarity) (const struct _RF_Similarity* self, const RF_String* str, double score_cutoff, double* similarity);

/* members */
    void* context;
} RF_Similarity;

typedef bool (*RF_SimilarityInit) (RF_Similarity* context, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings);

typedef struct {
    RF_KwargsInit kwargs_init;
    RF_SimilarityInit similarity_init;
} RfSimilarityFunctionTable;

typedef struct _RF_Distance {
    /**
     * @brief Destructor for RF_Distance
     * 
     * @param self pointer to RF_Distance instance to destruct
     */
    void (*dtor) (struct _RF_Distance* self);

    /**
     * @brief Calculate edit distance
     * 
     * @param[in] self pointer to RF_Distance instance
     * @param[in] str string to calculate distance with `strings` passed into `ctor`
     * @param[in] max argument for a score threshold between 0 and 100.
     * When the distance > max the distance is (size_t)-1.
     * @param[out] distance array of size `str_count` for results of the distance calculation
     * 
     * @return true on success and false with a Python exception set on failure
     */
    bool (*distance) (const struct _RF_Distance* self, const RF_String* str, size_t max, size_t* distance);

/* members */
    void* context;
} RF_Distance;

typedef bool (*RF_DistanceInit) (RF_Distance* context, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings);

typedef struct {
    RF_KwargsInit kwargs_init;
    RF_DistanceInit distance_init;
} RfDistanceFunctionTable;

#ifdef __cplusplus
}
#endif

#endif  /* RAPIDFUZZ_CAPI_H */
