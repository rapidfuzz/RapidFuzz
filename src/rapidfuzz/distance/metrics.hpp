#pragma once
#include "cpp_common.hpp"

/* Levenshtein */
static inline size_t levenshtein_distance_func(const RF_String& str1, const RF_String& str2,
                                                size_t insertion, size_t deletion, size_t substitution,
                                                size_t score_cutoff, size_t score_hint)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::levenshtein_distance(s1, s2, {insertion, deletion, substitution}, score_cutoff,
                                        score_hint);
    });
}

static inline bool LevenshteinDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                           const RF_String* str)
{
    rf::LevenshteinWeightTable weights = *static_cast<rf::LevenshteinWeightTable*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (weights.insert_cost == 1 && weights.delete_cost == 1 && weights.replace_cost == 1) {
        if (str_count != 1)
            return multi_distance_init<rf::experimental::MultiLevenshtein, size_t>(self, str_count, str);
    }
#endif

    return distance_init<rf::CachedLevenshtein, size_t>(self, str_count, str, weights);
}
static inline bool LevenshteinMultiStringSupport(const RF_Kwargs* kwargs)
{
    [[maybe_unused]] rf::LevenshteinWeightTable weights =
        *static_cast<rf::LevenshteinWeightTable*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (weights.insert_cost == 1 && weights.delete_cost == 1 && weights.replace_cost == 1) return true;
#endif
    return false;
}

static inline double levenshtein_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                          size_t insertion, size_t deletion,
                                                          size_t substitution, double score_cutoff,
                                                          double score_hint)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::levenshtein_normalized_distance(s1, s2, {insertion, deletion, substitution}, score_cutoff,
                                                   score_hint);
    });
}
static inline bool LevenshteinNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                     int64_t str_count, const RF_String* str)
{
    rf::LevenshteinWeightTable weights = *static_cast<rf::LevenshteinWeightTable*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (weights.insert_cost == 1 && weights.delete_cost == 1 && weights.replace_cost == 1) {
        if (str_count != 1)
            return multi_normalized_distance_init<rf::experimental::MultiLevenshtein, double>(self, str_count,
                                                                                              str);
    }
#endif

    return normalized_distance_init<rf::CachedLevenshtein, double>(self, str_count, str, weights);
}

static inline size_t levenshtein_similarity_func(const RF_String& str1, const RF_String& str2,
                                                  size_t insertion, size_t deletion, size_t substitution,
                                                  size_t score_cutoff, size_t score_hint)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::levenshtein_similarity(s1, s2, {insertion, deletion, substitution}, score_cutoff,
                                          score_hint);
    });
}

static inline bool LevenshteinSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                             const RF_String* str)
{
    rf::LevenshteinWeightTable weights = *static_cast<rf::LevenshteinWeightTable*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (weights.insert_cost == 1 && weights.delete_cost == 1 && weights.replace_cost == 1) {
        if (str_count != 1)
            return multi_similarity_init<rf::experimental::MultiLevenshtein, size_t>(self, str_count, str);
    }
#endif

    return similarity_init<rf::CachedLevenshtein, size_t>(self, str_count, str, weights);
}

static inline double levenshtein_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                            size_t insertion, size_t deletion,
                                                            size_t substitution, double score_cutoff,
                                                            double score_hint)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::levenshtein_normalized_similarity(s1, s2, {insertion, deletion, substitution},
                                                     score_cutoff, score_hint);
    });
}
static inline bool LevenshteinNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                       int64_t str_count, const RF_String* str)
{
    rf::LevenshteinWeightTable weights = *static_cast<rf::LevenshteinWeightTable*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (weights.insert_cost == 1 && weights.delete_cost == 1 && weights.replace_cost == 1) {
        if (str_count != 1)
            return multi_normalized_similarity_init<rf::experimental::MultiLevenshtein, double>(
                self, str_count, str);
    }
#endif

    return normalized_similarity_init<rf::CachedLevenshtein, double>(self, str_count, str, weights);
}

/* Damerau Levenshtein */
static inline size_t damerau_levenshtein_distance_func(const RF_String& str1, const RF_String& str2,
                                                        size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::experimental::damerau_levenshtein_distance(s1, s2, score_cutoff);
    });
}

static inline bool DamerauLevenshteinDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                  const RF_String* str)
{
    return distance_init<rf::experimental::CachedDamerauLevenshtein, size_t>(self, str_count, str);
}

static inline double damerau_levenshtein_normalized_distance_func(const RF_String& str1,
                                                                  const RF_String& str2, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::experimental::damerau_levenshtein_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool DamerauLevenshteinNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*,
                                                            int64_t str_count, const RF_String* str)
{
    return normalized_distance_init<rf::experimental::CachedDamerauLevenshtein, double>(self, str_count, str);
}

static inline size_t damerau_levenshtein_similarity_func(const RF_String& str1, const RF_String& str2,
                                                          size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::experimental::damerau_levenshtein_similarity(s1, s2, score_cutoff);
    });
}

static inline bool DamerauLevenshteinSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                    const RF_String* str)
{
    return similarity_init<rf::experimental::CachedDamerauLevenshtein, size_t>(self, str_count, str);
}

static inline double damerau_levenshtein_normalized_similarity_func(const RF_String& str1,
                                                                    const RF_String& str2,
                                                                    double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::experimental::damerau_levenshtein_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool DamerauLevenshteinNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*,
                                                              int64_t str_count, const RF_String* str)
{
    return normalized_similarity_init<rf::experimental::CachedDamerauLevenshtein, double>(self, str_count,
                                                                                          str);
}

/* Hamming */
static inline size_t hamming_distance_func(const RF_String& str1, const RF_String& str2, bool pad,
                                            size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::hamming_distance(s1, s2, pad, score_cutoff);
    });
}
static inline bool HammingDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                       const RF_String* str)
{
    bool pad = *static_cast<bool*>(kwargs->context);

    return distance_init<rf::CachedHamming, size_t>(self, str_count, str, pad);
}

static inline double hamming_normalized_distance_func(const RF_String& str1, const RF_String& str2, bool pad,
                                                      double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::hamming_normalized_distance(s1, s2, pad, score_cutoff);
    });
}
static inline bool HammingNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                 int64_t str_count, const RF_String* str)
{
    bool pad = *static_cast<bool*>(kwargs->context);

    return normalized_distance_init<rf::CachedHamming, double>(self, str_count, str, pad);
}

static inline size_t hamming_similarity_func(const RF_String& str1, const RF_String& str2, bool pad,
                                              size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::hamming_similarity(s1, s2, pad, score_cutoff);
    });
}
static inline bool HammingSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                         const RF_String* str)
{
    bool pad = *static_cast<bool*>(kwargs->context);

    return similarity_init<rf::CachedHamming, size_t>(self, str_count, str, pad);
}

static inline double hamming_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                        bool pad, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::hamming_normalized_similarity(s1, s2, pad, score_cutoff);
    });
}
static inline bool HammingNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                   int64_t str_count, const RF_String* str)
{
    bool pad = *static_cast<bool*>(kwargs->context);

    return normalized_similarity_init<rf::CachedHamming, double>(self, str_count, str, pad);
}

/* Indel */
static inline size_t indel_distance_func(const RF_String& str1, const RF_String& str2, size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::indel_distance(s1, s2, score_cutoff);
    });
}
static inline bool IndelDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                     const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_distance_init<rf::experimental::MultiIndel, size_t>(self, str_count, str);
#endif

    return distance_init<rf::CachedIndel, size_t>(self, str_count, str);
}
static inline bool IndelMultiStringSupport(const RF_Kwargs*)
{
#ifdef RAPIDFUZZ_X64
    return true;
#else
    return false;
#endif
}

static inline double indel_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                    double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::indel_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool IndelNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                               const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_distance_init<rf::experimental::MultiIndel, double>(self, str_count, str);
#endif

    return normalized_distance_init<rf::CachedIndel, double>(self, str_count, str);
}

static inline size_t indel_similarity_func(const RF_String& str1, const RF_String& str2,
                                            size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::indel_similarity(s1, s2, score_cutoff);
    });
}
static inline bool IndelSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                       const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_similarity_init<rf::experimental::MultiIndel, size_t>(self, str_count, str);
#endif

    return similarity_init<rf::CachedIndel, size_t>(self, str_count, str);
}

static inline double indel_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                      double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::indel_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool IndelNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                 const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_similarity_init<rf::experimental::MultiIndel, double>(self, str_count, str);
#endif

    return normalized_similarity_init<rf::CachedIndel, double>(self, str_count, str);
}

/* LCSseq */
static inline size_t lcs_seq_distance_func(const RF_String& str1, const RF_String& str2,
                                            size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::lcs_seq_distance(s1, s2, score_cutoff);
    });
}
static inline bool LCSseqDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                      const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_distance_init<rf::experimental::MultiLCSseq, size_t>(self, str_count, str);
#endif

    return distance_init<rf::CachedLCSseq, size_t>(self, str_count, str);
}
static inline bool LCSseqMultiStringSupport(const RF_Kwargs*)
{
#ifdef RAPIDFUZZ_X64
    return true;
#else
    return false;
#endif
}

static inline double lcs_seq_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                      double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::lcs_seq_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool LCSseqNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_distance_init<rf::experimental::MultiLCSseq, double>(self, str_count, str);
#endif

    return normalized_distance_init<rf::CachedLCSseq, double>(self, str_count, str);
}

static inline size_t lcs_seq_similarity_func(const RF_String& str1, const RF_String& str2,
                                              size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::lcs_seq_similarity(s1, s2, score_cutoff);
    });
}
static inline bool LCSseqSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                        const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_similarity_init<rf::experimental::MultiLCSseq, size_t>(self, str_count, str);
#endif

    return similarity_init<rf::CachedLCSseq, size_t>(self, str_count, str);
}

static inline double lcs_seq_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                        double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::lcs_seq_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool LCSseqNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                  const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_similarity_init<rf::experimental::MultiLCSseq, double>(self, str_count, str);
#endif

    return normalized_similarity_init<rf::CachedLCSseq, double>(self, str_count, str);
}

static inline rf::Editops hamming_editops_func(const RF_String& str1, const RF_String& str2, bool pad)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::hamming_editops(s1, s2, pad);
    });
}

static inline rf::Editops levenshtein_editops_func(const RF_String& str1, const RF_String& str2,
                                                   size_t score_hint)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::levenshtein_editops(s1, s2, score_hint);
    });
}

static inline rf::Editops indel_editops_func(const RF_String& str1, const RF_String& str2)
{
    return visitor(str1, str2, [](auto s1, auto s2) {
        return rf::indel_editops(s1, s2);
    });
}

static inline rf::Editops lcs_seq_editops_func(const RF_String& str1, const RF_String& str2)
{
    return visitor(str1, str2, [](auto s1, auto s2) {
        return rf::lcs_seq_editops(s1, s2);
    });
}

/* Damerau Levenshtein */
static inline size_t osa_distance_func(const RF_String& str1, const RF_String& str2, size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::osa_distance(s1, s2, score_cutoff);
    });
}

static inline bool OSADistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                   const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1) return multi_distance_init<rf::experimental::MultiOSA, size_t>(self, str_count, str);
#endif

    return distance_init<rf::CachedOSA, size_t>(self, str_count, str);
}

static inline bool OSAMultiStringSupport(const RF_Kwargs*)
{
#ifdef RAPIDFUZZ_X64
    return true;
#else
    return false;
#endif
}

static inline double osa_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                  double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::osa_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool OSANormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                             const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_distance_init<rf::experimental::MultiOSA, double>(self, str_count, str);
#endif

    return normalized_distance_init<rf::CachedOSA, double>(self, str_count, str);
}

static inline size_t osa_similarity_func(const RF_String& str1, const RF_String& str2, size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::osa_similarity(s1, s2, score_cutoff);
    });
}

static inline bool OSASimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                     const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_similarity_init<rf::experimental::MultiOSA, size_t>(self, str_count, str);
#endif

    return similarity_init<rf::CachedOSA, size_t>(self, str_count, str);
}

static inline double osa_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                    double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::osa_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool OSANormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                               const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_similarity_init<rf::experimental::MultiOSA, double>(self, str_count, str);
#endif

    return normalized_similarity_init<rf::CachedOSA, double>(self, str_count, str);
}

/* Jaro */
static inline double jaro_distance_func(const RF_String& str1, const RF_String& str2, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_distance(s1, s2, score_cutoff);
    });
}
static inline bool JaroDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                    const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1) return multi_distance_init<rf::experimental::MultiJaro, double>(self, str_count, str);
#endif

    return distance_init<rf::CachedJaro, double>(self, str_count, str);
}

static inline double jaro_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                   double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool JaroNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                              const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_distance_init<rf::experimental::MultiJaro, double>(self, str_count, str);
#endif

    return normalized_distance_init<rf::CachedJaro, double>(self, str_count, str);
}

static inline double jaro_similarity_func(const RF_String& str1, const RF_String& str2, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_similarity(s1, s2, score_cutoff);
    });
}
static inline bool JaroSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                      const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_similarity_init<rf::experimental::MultiJaro, double>(self, str_count, str);
#endif

    return similarity_init<rf::CachedJaro, double>(self, str_count, str);
}

static inline double jaro_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                     double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool JaroNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                const RF_String* str)
{
#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_similarity_init<rf::experimental::MultiJaro, double>(self, str_count, str);
#endif

    return normalized_similarity_init<rf::CachedJaro, double>(self, str_count, str);
}

static inline bool JaroMultiStringSupport(const RF_Kwargs*)
{
#ifdef RAPIDFUZZ_X64
    return true;
#else
    return false;
#endif
}

/* JaroWinkler */
static inline double jaro_winkler_distance_func(const RF_String& str1, const RF_String& str2,
                                                double prefix_weight, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_winkler_distance(s1, s2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                           const RF_String* str)
{
    double prefix_weight = *static_cast<double*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_distance_init<rf::experimental::MultiJaroWinkler, double>(self, str_count, str,
                                                                               prefix_weight);
#endif

    return distance_init<rf::CachedJaroWinkler, double>(self, str_count, str, prefix_weight);
}

static inline double jaro_winkler_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                           double prefix_weight, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_winkler_normalized_distance(s1, s2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                     int64_t str_count, const RF_String* str)
{
    double prefix_weight = *static_cast<double*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_distance_init<rf::experimental::MultiJaroWinkler, double>(self, str_count,
                                                                                          str, prefix_weight);
#endif

    return normalized_distance_init<rf::CachedJaroWinkler, double>(self, str_count, str, prefix_weight);
}

static inline double jaro_winkler_similarity_func(const RF_String& str1, const RF_String& str2,
                                                  double prefix_weight, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_winkler_similarity(s1, s2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs, int64_t str_count,
                                             const RF_String* str)
{
    double prefix_weight = *static_cast<double*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_similarity_init<rf::experimental::MultiJaroWinkler, double>(self, str_count, str,
                                                                                 prefix_weight);
#endif

    return similarity_init<rf::CachedJaroWinkler, double>(self, str_count, str, prefix_weight);
}

static inline double jaro_winkler_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                             double prefix_weight, double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::jaro_winkler_normalized_similarity(s1, s2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs* kwargs,
                                                       int64_t str_count, const RF_String* str)
{
    double prefix_weight = *static_cast<double*>(kwargs->context);

#ifdef RAPIDFUZZ_X64
    if (str_count != 1)
        return multi_normalized_similarity_init<rf::experimental::MultiJaroWinkler, double>(
            self, str_count, str, prefix_weight);
#endif

    return normalized_similarity_init<rf::CachedJaroWinkler, double>(self, str_count, str, prefix_weight);
}

static inline bool JaroWinklerMultiStringSupport(const RF_Kwargs*)
{
#ifdef RAPIDFUZZ_X64
    return true;
#else
    return false;
#endif
}

/* Prefix */
static inline size_t prefix_distance_func(const RF_String& str1, const RF_String& str2, size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::prefix_distance(s1, s2, score_cutoff);
    });
}
static inline bool PrefixDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                      const RF_String* str)
{
    return distance_init<rf::CachedPrefix, size_t>(self, str_count, str);
}

static inline double prefix_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                     double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::prefix_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool PrefixNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                const RF_String* str)
{
    return normalized_distance_init<rf::CachedPrefix, double>(self, str_count, str);
}

static inline size_t prefix_similarity_func(const RF_String& str1, const RF_String& str2,
                                             size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::prefix_similarity(s1, s2, score_cutoff);
    });
}
static inline bool PrefixSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                        const RF_String* str)
{
    return similarity_init<rf::CachedPrefix, size_t>(self, str_count, str);
}

static inline double prefix_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                       double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::prefix_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool PrefixNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                  const RF_String* str)
{
    return normalized_similarity_init<rf::CachedPrefix, double>(self, str_count, str);
}

/* Postfix */
static inline size_t postfix_distance_func(const RF_String& str1, const RF_String& str2,
                                            size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::postfix_distance(s1, s2, score_cutoff);
    });
}
static inline bool PostfixDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                       const RF_String* str)
{
    return distance_init<rf::CachedPostfix, size_t>(self, str_count, str);
}

static inline double postfix_normalized_distance_func(const RF_String& str1, const RF_String& str2,
                                                      double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::postfix_normalized_distance(s1, s2, score_cutoff);
    });
}
static inline bool PostfixNormalizedDistanceInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                 const RF_String* str)
{
    return normalized_distance_init<rf::CachedPostfix, double>(self, str_count, str);
}

static inline size_t postfix_similarity_func(const RF_String& str1, const RF_String& str2,
                                              size_t score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::postfix_similarity(s1, s2, score_cutoff);
    });
}
static inline bool PostfixSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                         const RF_String* str)
{
    return similarity_init<rf::CachedPostfix, size_t>(self, str_count, str);
}

static inline double postfix_normalized_similarity_func(const RF_String& str1, const RF_String& str2,
                                                        double score_cutoff)
{
    return visitor(str1, str2, [&](auto s1, auto s2) {
        return rf::postfix_normalized_similarity(s1, s2, score_cutoff);
    });
}
static inline bool PostfixNormalizedSimilarityInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count,
                                                   const RF_String* str)
{
    return normalized_similarity_init<rf::CachedPostfix, double>(self, str_count, str);
}
