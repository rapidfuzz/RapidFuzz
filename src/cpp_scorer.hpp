#pragma once
#include "cpp_common.hpp"

template <typename CachedScorer>
static void scorer_deinit(RF_ScorerFunc* self)
{
    delete (CachedScorer*)self->context;
}

template<typename CachedScorer>
static inline bool scorer_func_wrapper_f64(const RF_ScorerFunc* self, const RF_String* str, double score_cutoff, double* result)
{
    CachedScorer& scorer = *(CachedScorer*)self->context;
    try {
        *result = visit(*str, [&](auto s){
            return scorer.ratio(s, score_cutoff);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<typename CachedScorer>
static inline bool scorer_func_wrapper_u64(const RF_ScorerFunc* self, const RF_String* str, uint64_t score_cutoff, uint64_t* result)
{
    CachedScorer& scorer = *(CachedScorer*)self->context;
    try {
        *result = visit(*str, [&](auto s){
            return scorer.distance(s, score_cutoff);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}


template<template <typename> class CachedScorer, typename Sentence, typename ...Args>
static inline RF_ScorerFunc get_ScorerContext_f64(Sentence str, Args... args)
{
    RF_ScorerFunc context;
    context.context = (void*) new CachedScorer<Sentence>(str, args...);

    context.call.f64 = scorer_func_wrapper_f64<CachedScorer<Sentence>>;
    context.dtor = scorer_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename Sentence, typename ...Args>
static inline RF_ScorerFunc get_ScorerContext_u64(Sentence str, Args... args)
{
    RF_ScorerFunc context;
    context.context = (void*) new CachedScorer<Sentence>(str, args...);

    context.call.u64 = scorer_func_wrapper_u64<CachedScorer<Sentence>>;
    context.dtor = scorer_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline bool scorer_init_f64(RF_ScorerFunc* self, size_t str_count, const RF_String* strings, Args... args)
{
    try {
        /* todo support different string counts, which is required e.g. for SIMD */
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto s){
            return get_ScorerContext_f64<CachedScorer>(s, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline bool scorer_init_u64(RF_ScorerFunc* self, size_t str_count, const RF_String* strings, Args... args)
{
    try {
        /* todo support different string counts, which is required e.g. for SIMD */
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto s){
            return get_ScorerContext_u64<CachedScorer>(s, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

/* ratio */

static inline double ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
static inline bool RatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedRatio>(self, str_count, str);
}

static inline double partial_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
static inline bool PartialRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedPartialRatio>(self, str_count, str);
}


static inline double token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline bool TokenSortRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedTokenSortRatio>(self, str_count, str);
}


static inline double token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline bool TokenSetRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedTokenSetRatio>(self, str_count, str);
}

static inline double token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
static inline bool TokenRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedTokenRatio>(self, str_count, str);
}

static inline double partial_token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline bool PartialTokenSortRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedPartialTokenSortRatio>(self, str_count, str);
}


static inline double partial_token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline bool PartialTokenSetRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedPartialTokenSetRatio>(self, str_count, str);
}

static inline double partial_token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
static inline bool PartialTokenRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedPartialTokenRatio>(self, str_count, str);
}

static inline double WRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
static inline bool WRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedWRatio>(self, str_count, str);
}

static inline double QRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
static inline bool QRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<fuzz::CachedQRatio>(self, str_count, str);
}

/* string_metric */
static inline size_t levenshtein_func(const RF_String& s1, const RF_String& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
}
static inline bool LevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_u64<string_metric::CachedLevenshtein>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline double normalized_levenshtein_func(const RF_String& s1, const RF_String& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}
static inline bool NormalizedLevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedNormalizedLevenshtein>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline size_t hamming_func(const RF_String& s1, const RF_String& s2, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
}
static inline bool HammingInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_u64<string_metric::CachedHamming>(self, str_count, str);
}

static inline double normalized_hamming_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
static inline bool NormalizedHammingInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedNormalizedHamming>(self, str_count, str);
}

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedJaroSimilarity>(self, str_count, str);
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return scorer_init_f64<string_metric::CachedJaroWinklerSimilarity>(self, str_count, str, *(double*)(kwargs->context));
}

static inline std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}
