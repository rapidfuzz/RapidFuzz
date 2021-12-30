#ifndef RAPIDFUZZ_CAPI_H
#define RAPIDFUZZ_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

enum RF_StringType {
    RF_UINT8,  /* uint8_t */
    RF_UINT16, /* uint16_t */
    RF_UINT32, /* uint32_t */
    RF_UINT64  /* uint64_t */
};

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

/**
 * @brief convert python object to RF_String and preprocess it
 *
 * @param[in] obj Python object
 * @param[out] str Preprocessed String
 *
 * @return true on success and false with a Python exception set on failure
 */
typedef bool (*RF_Preprocess) (PyObject* obj, RF_String* str);

/**
 * @brief struct describing a Processor callback function.
 */
typedef struct {
#define PREPROCESSOR_STRUCT_VERSION ((uint32_t)1)
    uint32_t version; /**< version number of the structure. Set to PREPROCESSOR_STRUCT_VERSION */
    RF_Preprocess preprocess;
} RF_Preprocessor;


typedef struct _RF_Kwargs {
    /**
     * @brief destructor for RF_Kwargs
     *
     * @param self pointer to RF_Kwargs instance to destruct
     */
    void (*dtor) (struct _RF_Kwargs* self);

/* members */
    void* context;
} RF_Kwargs;

/**
 * @brief construct RF_Kwargs
 *
 * @param[out] self constructed RF_Kwargs instance
 * @param[in] kwargs Python dictionary holding keyword arguments
 *
 * @return true on success and false with a Python exception set on failure
 */
typedef bool (*RF_KwargsInit) (RF_Kwargs* self, PyObject* kwargs);


typedef struct _RF_ScorerFunc {
    /**
     * @brief Destructor for RF_ScorerFunc
     *
     * @param self pointer to RF_ScorerFunc instance to destruct
     */
    void (*dtor) (struct _RF_ScorerFunc* self);

    /**
     * @brief Calculate edit distance
     *
     * @note has to be specified using RF_SCORER_FLAG_*:
     * - RF_SCORER_FLAG_RESULT_F64 -> call_f64
     * - RF_SCORER_FLAG_RESULT_I64 -> call_i64
     * - RF_SCORER_FLAG_RESULT_U64 -> call_u64
     * 
     * @param[in] self pointer to RF_ScorerFunc instance
     * @param[in] str string to calculate distance with `strings` passed into `ctor`
     * @param[in] score_cutoff argument for a score threshold
     * @param[out] result array of size `str_count` for results of the calculation
     *
     * @return true on success and false with a Python exception set on failure
     */
    union {
        bool (*f64) (const struct _RF_ScorerFunc* self, const RF_String* str, double score_cutoff, double* result);
        bool (*u64) (const struct _RF_ScorerFunc* self, const RF_String* str, uint64_t score_cutoff, uint64_t* result);
        bool (*i64) (const struct _RF_ScorerFunc* self, const RF_String* str, int64_t score_cutoff, int64_t* result);
    } call;

/* members */
    void* context;
} RF_ScorerFunc;

/**
 * @brief construct RF_ScorerFunc.
 *
 * @param[out] self constructed RF_ScorerFunc instance
 * @param[in] kwargs keyword arguments for additional parameters
 * @param[in] str_count size of the strings array can only be != 1 if
 *                      RF_SCORER_FLAG_MULTI_STRING is set
 * @param[in] strings array of strings to compare in distance function
 *
 * @return true on success and false with a Python exception set on failure
 */
typedef bool (*RF_ScorerFuncInit) (RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* strings);

/* scorer supports str_count != 1.
 * This is useful for scorers which have SIMD support
 */
#define RF_SCORER_FLAG_MULTI_STRING          ((uint32_t)1 << 0)

/* scorer returns result as double */
#define RF_SCORER_FLAG_RESULT_F64            ((uint32_t)1 << 1)

/* scorer returns result as int64_t */
#define RF_SCORER_FLAG_RESULT_I64            ((uint32_t)1 << 2)

/* scorer returns result as uint64_t */
#define RF_SCORER_FLAG_RESULT_U64            ((uint32_t)1 << 3)

/* scorer is symmetric: scorer(a, b) == scorer(b, a) */
#define RF_SCORER_FLAG_SYMMETRIC             ((uint32_t)1 << 11)

/* scorer adheres to triangle inequality: scorer(a,b) <= scorer(a,c) + scorer(b,c)
 * Implies that the scorer is symmetric
 */
#define RF_SCORER_FLAG_TRIANGLE_INEQUALITY   ((uint32_t)1 << 12 | RF_SCORER_FLAG_SYMMETRIC)

typedef struct _RF_ScorerFlags {
    uint32_t flags;
    /**
     * @brief optimal score which can be achieved.
     */
    union {
        double   f64;
        uint64_t u64;
        int64_t  i64;
    } optimal_score;

    /**
     * @brief worst score which can be achieved.
     */
    union {
        double   f64;
        uint64_t u64;
        int64_t  i64;
    } worst_score;
} RF_ScorerFlags;

/**
 * @brief retrieve flags associated with the scorer
 *
 * @param[in] kwargs keyword arguments of the scorer
 * @param[out] scorer_flags Scorer Flags associated with the scorer
 *
 * @return true on success and false with a Python exception set on failure
 */
typedef bool (*RF_GetScorerFlags) (const RF_Kwargs* kwargs, RF_ScorerFlags* scorer_flags);

/**
 * @brief struct describing a Scorer callback function.
 */
typedef struct {
#define SCORER_STRUCT_VERSION ((uint32_t)1)
    uint32_t version; /**< version number of the structure. Set to SCORER_STRUCT_VERSION */
    RF_KwargsInit kwargs_init;
    RF_GetScorerFlags get_scorer_flags;
    RF_ScorerFuncInit scorer_func_init;
} RF_Scorer;

#ifdef __cplusplus
}
#endif

#endif  /* RAPIDFUZZ_CAPI_H */
