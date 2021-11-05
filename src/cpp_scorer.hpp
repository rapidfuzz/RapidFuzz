#pragma once
#include "cpp_common.hpp"

template <typename CachedScorer>
static void similarity_deinit(RF_Similarity* self)
{
    delete (CachedScorer*)self->context;
}

template <typename CachedScorer>
static void distance_deinit(RF_Distance* self)
{
    delete (CachedScorer*)self->context;
}

template<typename CachedScorer>
static inline bool similarity_func_wrapper(const RF_Similarity* self, const RF_String* str, double score_cutoff, double* similarity)
{
    try {
        *similarity = visit(*str, [&](auto s){
            return ((CachedScorer*)self->context)->ratio(s, score_cutoff);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<typename CachedDistance>
static inline bool distance_func_wrapper(const RF_Distance* self, const RF_String* str, size_t max, size_t* distance)
{
    try {
        *distance = visit(*str, [&](auto s){
            return ((CachedDistance*)self->context)->distance(s, max);
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
static inline RF_Similarity get_SimilarityContext(Sentence str, Args... args)
{
    RF_Similarity context;
    context.context = (void*) new CachedScorer<Sentence>(str, args...);

    context.similarity = similarity_func_wrapper<CachedScorer<Sentence>>;
    context.dtor = similarity_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedDistance, typename Sentence, typename ...Args>
static inline RF_Distance get_DistanceContext(Sentence str, Args... args)
{
    RF_Distance context;
    context.context = (void*) new CachedDistance<Sentence>(str, args...);

    context.distance = distance_func_wrapper<CachedDistance<Sentence>>;
    context.dtor = distance_deinit<CachedDistance<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline bool similarity_init(RF_Similarity* self, size_t str_count, const RF_String* strings, Args... args)
{
    try {
        /* todo support different string counts, which is required e.g. for SIMD */
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto s){
            return get_SimilarityContext<CachedScorer>(s, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

template<template <typename> class CachedDistance, typename ...Args>
static inline bool distance_init(RF_Distance* self, size_t str_count, const RF_String* strings, Args... args)
{
    try {
        /* todo support different string counts, which is required e.g. for SIMD */
        if (str_count != 1)
        {
            throw std::logic_error("Only str_count == 1 supported");
        }
        *self = visit(*strings, [&](auto s){
            return get_DistanceContext<CachedDistance>(s, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return false;
    }
    return true;
}

static inline RF_Scorer CreateSimilarity(RF_KwargsInit kwargs_init, RF_SimilarityInit similarity_init)
{
    RF_Scorer scorer{};
    scorer.scorer_type = RF_SIMILARITY;
    scorer.kwargs_init = kwargs_init;
    scorer.scorer.similarity_init = similarity_init;
    return scorer;
}

static inline RF_Scorer CreateDistance(RF_KwargsInit kwargs_init, RF_DistanceInit distance_init)
{
    RF_Scorer scorer{};
    scorer.scorer_type = RF_DISTANCE;
    scorer.kwargs_init = kwargs_init;
    scorer.scorer.distance_init = distance_init;
    return scorer;
}

/* ratio */

static inline double ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedRatio>(self, str_count, str);
        }
    );
}

static inline double partial_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreatePartialRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedPartialRatio>(self, str_count, str);
        }
    );
}

static inline double token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateTokenSortRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedTokenSortRatio>(self, str_count, str);
        }
    );
}

static inline double token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateTokenSetRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedTokenSetRatio>(self, str_count, str);
        }
    );
}

static inline double token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateTokenRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedTokenRatio>(self, str_count, str);
        }
    );
}

static inline double partial_token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreatePartialTokenSortRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedPartialTokenSortRatio>(self, str_count, str);
        }
    );
}

static inline double partial_token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreatePartialTokenSetRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedPartialTokenSetRatio>(self, str_count, str);
        }
    );
}

static inline double partial_token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreatePartialTokenRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedPartialTokenRatio>(self, str_count, str);
        }
    );
}

static inline double WRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateWRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedWRatio>(self, str_count, str);
        }
    );
}

static inline double QRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateQRatioFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<fuzz::CachedQRatio>(self, str_count, str);
        }
    );
}

/* string_metric */
static inline PyObject* levenshtein_func(const RF_String& s1, const RF_String& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}
static inline bool LevenshteinInit(RF_Distance* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return distance_init<string_metric::CachedLevenshtein>(
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
static inline bool NormalizedLevenshteinInit(RF_Similarity* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return similarity_init<string_metric::CachedNormalizedLevenshtein>(
        self, str_count, str, *(rapidfuzz::LevenshteinWeightTable*)(kwargs->context)
    );
}

static inline PyObject* hamming_func(const RF_String& s1, const RF_String& s2, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}
static inline RF_Scorer CreateHammingFunctionTable()
{
    return CreateDistance(
        nullptr,
        [](RF_Distance* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return distance_init<string_metric::CachedHamming>(self, str_count, str);
        }
    );
}

static inline double normalized_hamming_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateNormalizedHammingFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<string_metric::CachedNormalizedHamming>(self, str_count, str);
        }
    );
}

static inline double jaro_similarity_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
static inline RF_Scorer CreateJaroSimilarityFunctionTable()
{
    return CreateSimilarity(
        nullptr,
        [](RF_Similarity* self, const RF_Kwargs*, size_t str_count, const RF_String* str) {
            return similarity_init<string_metric::CachedJaroSimilarity>(self, str_count, str);
        }
    );
}

static inline double jaro_winkler_similarity_func(const RF_String& s1, const RF_String& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
static inline bool JaroWinklerSimilarityInit(RF_Similarity* self, const RF_Kwargs* kwargs, size_t str_count, const RF_String* str)
{
    return similarity_init<string_metric::CachedJaroWinklerSimilarity>(self, str_count, str, *(double*)(kwargs->context));
}

static inline std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_func(
    const RF_String& s1, const RF_String& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}
