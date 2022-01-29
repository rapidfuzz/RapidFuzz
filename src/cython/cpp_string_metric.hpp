#pragma once
#include "cpp_common.hpp"
#include <jaro_winkler/jaro_winkler.hpp>

template<typename CachedScorer, typename T>
static inline bool legacy_normalized_similarity_func_wrapper(const RF_ScorerFunc* self, const RF_String* str, int64_t str_count, T score_cutoff, T* result)
{
    CachedScorer& scorer = *(CachedScorer*)self->context;
    try {
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *result = visit(*str, [&](auto first, auto last){
            return scorer.normalized_similarity(first, last, score_cutoff) * 100.0;
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<template <typename> class CachedScorer, typename T, typename InputIt1, typename ...Args>
static inline RF_ScorerFunc legacy_get_ScorerContext_normalized_similarity(InputIt1 first1, InputIt1 last1, Args... args)
{
    using CharT1 = typename std::iterator_traits<InputIt1>::value_type;
    RF_ScorerFunc context;
    context.context = (void*) new CachedScorer<CharT1>(first1, last1, args...);

    assign_callback(context, legacy_normalized_similarity_func_wrapper<CachedScorer<CharT1>, T>);
    context.dtor = scorer_deinit<CachedScorer<CharT1>>;
    return context;
}

template<template <typename> class CachedScorer, typename T, typename ...Args>
static inline bool legacy_normalized_similarity_init(RF_ScorerFunc* self, int64_t str_count, const RF_String* strings, Args... args)
{
    try {
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto first1, auto last1){
            return legacy_get_ScorerContext_normalized_similarity<CachedScorer, T>(first1, last1, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

static inline int64_t levenshtein_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_distance(first1, last1, first2, last2, {insertion, deletion, substitution}, max);
    });
}
static inline bool LevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return distance_init<rapidfuzz::CachedLevenshtein, int64_t>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline double normalized_levenshtein_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_normalized_similarity(
            first1, last1, first2, last2, {insertion, deletion, substitution}, score_cutoff / 100
        ) * 100;
    });
}
static inline bool NormalizedLevenshteinInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return legacy_normalized_similarity_init<rapidfuzz::CachedLevenshtein, double>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline int64_t hamming_func(const RF_String& s1, const RF_String& s2, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_distance(first1, last1, first2, last2, max);
    });
}
static inline bool HammingInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return distance_init<rapidfuzz::CachedHamming, int64_t>(self, str_count, str);
}

static inline double normalized_hamming_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_normalized_distance(first1, last1, first2, last2, score_cutoff / 100) * 100;
    });
}
static inline bool NormalizedHammingInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return legacy_normalized_similarity_init<rapidfuzz::CachedHamming, double>(self, str_count, str);
}

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_similarity(first1, last1, first2, last2, score_cutoff / 100) * 100;
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return legacy_normalized_similarity_init<jaro_winkler::CachedJaroSimilarity, double>(self, str_count, str);
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_winkler_similarity(first1, last1, first2, last2, prefix_weight, score_cutoff / 100) * 100;
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return legacy_normalized_similarity_init<jaro_winkler::CachedJaroWinklerSimilarity, double>(self, str_count, str, *(double*)(kwargs->context));
}

static inline rapidfuzz::Editops levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_editops(first1, last1, first2, last2);
    });
}
