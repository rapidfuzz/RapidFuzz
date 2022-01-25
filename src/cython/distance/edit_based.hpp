#pragma once
#include "cpp_common.hpp"
#include <iostream>
#include <jaro_winkler/jaro_winkler.hpp>

/* Levenshtein */
static inline int64_t levenshtein_distance_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_distance(first1, last1, first2, last2, {insertion, deletion, substitution}, max);
    });
}

static inline bool LevenshteinDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return distance_init<rapidfuzz::CachedLevenshtein, int64_t>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline double levenshtein_normalized_distance_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_normalized_distance(
            first1, last1, first2, last2, {insertion, deletion, substitution}, score_cutoff
        );
    });
}
static inline bool LevenshteinNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return normalized_distance_init<rapidfuzz::CachedLevenshtein, double>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline int64_t levenshtein_similarity_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_similarity(first1, last1, first2, last2, {insertion, deletion, substitution}, max);
    });
}

static inline bool LevenshteinSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return similarity_init<rapidfuzz::CachedLevenshtein, int64_t>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline double levenshtein_normalized_similarity_func(const RF_String& s1, const RF_String& s2,
    int64_t insertion, int64_t deletion, int64_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_normalized_similarity(
            first1, last1, first2, last2, {insertion, deletion, substitution}, score_cutoff
        );
    });
}
static inline bool LevenshteinNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return normalized_similarity_init<rapidfuzz::CachedLevenshtein, double>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

/* Hamming */
static inline int64_t hamming_distance_func(const RF_String& s1, const RF_String& s2, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_distance(first1, last1, first2, last2, max);
    });
}
static inline bool HammingDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return distance_init<rapidfuzz::CachedHamming, int64_t>(self, str_count, str);
}

static inline double hamming_normalized_distance_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_normalized_distance(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool HammingNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return normalized_distance_init<rapidfuzz::CachedHamming, double>(self, str_count, str);
}

static inline int64_t hamming_similarity_func(const RF_String& s1, const RF_String& s2, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_similarity(first1, last1, first2, last2, max);
    });
}
static inline bool HammingSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<rapidfuzz::CachedHamming, int64_t>(self, str_count, str);
}

static inline double hamming_normalized_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::hamming_normalized_similarity(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool HammingNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return normalized_similarity_init<rapidfuzz::CachedHamming, double>(self, str_count, str);
}

/* Indel */
static inline int64_t indel_distance_func(const RF_String& s1, const RF_String& s2, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::indel_distance(first1, last1, first2, last2, max);
    });
}
static inline bool IndelDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return distance_init<rapidfuzz::CachedIndel, int64_t>(self, str_count, str);
}

static inline double indel_normalized_distance_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::indel_normalized_distance(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool IndelNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return normalized_distance_init<rapidfuzz::CachedIndel, double>(self, str_count, str);
}

static inline int64_t indel_similarity_func(const RF_String& s1, const RF_String& s2, int64_t max)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::indel_similarity(first1, last1, first2, last2, max);
    });
}
static inline bool IndelSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<rapidfuzz::CachedIndel, int64_t>(self, str_count, str);
}

static inline double indel_normalized_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::indel_normalized_similarity(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool IndelNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return normalized_similarity_init<rapidfuzz::CachedIndel, double>(self, str_count, str);
}

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_similarity(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<jaro_winkler::CachedJaroSimilarity, double>(self, str_count, str);
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return jaro_winkler::jaro_winkler_similarity(first1, last1, first2, last2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count, const RF_String* str)
{
    return similarity_init<jaro_winkler::CachedJaroWinklerSimilarity, double>(self, str_count, str, *(double*)(kwargs->context));
}

static inline rapidfuzz::Editops levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::levenshtein_editops(first1, last1, first2, last2);
    });
}

static inline rapidfuzz::Editops indel_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto first1, auto last1, auto first2, auto last2) {
        return rapidfuzz::indel_editops(first1, last1, first2, last2);
    });
}
