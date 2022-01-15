#pragma once
#include "cpp_common.hpp"
#include <iostream>

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
        return string_metric::normalized_levenshtein(
            str1, str2, {insertion, deletion, substitution}, score_cutoff
        );
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

static inline size_t indel_func(const RF_String& s1, const RF_String& s2, size_t max)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {1, 1, 2}, max);
    });
}
static inline bool IndelInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    rapidfuzz::LevenshteinWeightTable weights = {1, 1, 2};
    return scorer_init_u64<string_metric::CachedLevenshtein>(self, str_count, str, weights);
}

static inline double normalized_indel_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {1, 1, 2}, score_cutoff);
    });
}
static inline bool NormalizedIndelInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    rapidfuzz::LevenshteinWeightTable weights = {1, 1, 2};
    return scorer_init_f64<string_metric::CachedNormalizedLevenshtein>(self, str_count, str, weights);
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

static inline rapidfuzz::Editops levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}

static inline rapidfuzz::Editops llcs_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::llcs_editops(str1, str2);
    });
}
