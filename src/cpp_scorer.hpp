#pragma once
#include "cpp_common.hpp"

template <typename CachedScorer>
static void cached_deinit(RfSimilarityContext* context)
{
    delete (CachedScorer*)context->context;
}

template<typename CachedScorer>
static inline double similarity_func_wrapper(const RfSimilarityContext* context, const RfString* str, double score_cutoff)
{
    return visit(*str, [&](auto s){
        return ((CachedScorer*)context->context)->ratio(s, score_cutoff);
    });
}

template<template <typename> class CachedScorer, typename Sentence, typename ...Args>
static inline RfSimilarityContext get_SimilarityContext(Sentence str, Args... args)
{
    RfSimilarityContext context;
    context.context = (void*) new CachedScorer<Sentence>(str, args...);

    context.similarity = similarity_func_wrapper<CachedScorer<Sentence>>;
    context.deinit = cached_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline RfSimilarityContext similarity_init(const RfString* str, Args... args)
{
    return visit(*str, [&](auto s){
        return get_SimilarityContext<CachedScorer>(s, args...);
    });
}

/* ratio */

double ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
double ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedRatio>(str);
        }
    };
}

double partial_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
double partial_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreatePartialRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialRatio>(str);
        }
    };
}

double token_sort_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
double token_sort_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenSortRatio>(str);
        }
    };
}

double token_set_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
double token_set_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenSetRatio>(str);
        }
    };
}

double token_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
double token_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenRatio>(str);
        }
    };
}

double partial_token_sort_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_sort_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreatePartialTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenSortRatio>(str);
        }
    };
}

double partial_token_set_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_set_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreatePartialTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenSetRatio>(str);
        }
    };
}

double partial_token_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
double partial_token_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreatePartialTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenRatio>(str);
        }
    };
}

double WRatio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
double WRatio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateWRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedWRatio>(str);
        }
    };
}

double QRatio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
double QRatio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
static RfSimilarityFunctionTable CreateQRatioFunctionTable()
{
    return {
        nullptr,
        [](const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedQRatio>(str);
        }
    };
}

/* string_metric */
PyObject* levenshtein_no_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}
PyObject* levenshtein_default_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}

double normalized_levenshtein_no_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}
double normalized_levenshtein_default_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}

PyObject* hamming_no_process(const RfString& s1, const RfString& s2, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}
PyObject* hamming_default_process(const RfString& s1, const RfString& s2, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}

double normalized_hamming_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
double normalized_hamming_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}

double jaro_similarity_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
double jaro_similarity_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}

double jaro_winkler_similarity_no_process(const RfString& s1, const RfString& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
double jaro_winkler_similarity_default_process(const RfString& s1, const RfString& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}

std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_no_process(
    const RfString& s1, const RfString& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}

std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_default_process(
    const RfString& s1, const RfString& s2)
{
    return visitor_default_process(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });     
}
