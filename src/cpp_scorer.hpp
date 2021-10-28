#pragma once
#include "cpp_common.hpp"

template <typename CachedScorer>
static void cached_deinit(RfSimilarityContext* context)
{
    delete (CachedScorer*)context->context;
}

template<typename CachedScorer>
static inline int similarity_func_wrapper(double* similarity, const RfSimilarityContext* context, const RfString* str, double score_cutoff)
{
    try {
        *similarity = visit(*str, [&](auto s){
            return ((CachedScorer*)context->context)->ratio(s, score_cutoff);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return -1;
    }
    return 0;
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
static inline int similarity_init(RfSimilarityContext* context, const RfString* str, Args... args)
{
    try {
        *context = visit(*str, [&](auto s){
            return get_SimilarityContext<CachedScorer>(s, args...);
        });
    } catch(...) {
      PyGILState_STATE gilstate_save = PyGILState_Ensure();
      CppExn2PyErr();
      PyGILState_Release(gilstate_save);
      return -1;
    }
    return 0;
}

/* ratio */

static inline double ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
static inline double ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedRatio>(context, str);
        }
    };
}

static inline double partial_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
static inline double partial_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreatePartialRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialRatio>(context, str);
        }
    };
}

static inline double token_sort_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline double token_sort_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenSortRatio>(context, str);
        }
    };
}

static inline double token_set_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline double token_set_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenSetRatio>(context, str);
        }
    };
}

static inline double token_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
static inline double token_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::token_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedTokenRatio>(context, str);
        }
    };
}

static inline double partial_token_sort_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline double partial_token_sort_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_sort_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreatePartialTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenSortRatio>(context, str);
        }
    };
}

static inline double partial_token_set_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline double partial_token_set_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_set_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreatePartialTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenSetRatio>(context, str);
        }
    };
}

static inline double partial_token_ratio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
static inline double partial_token_ratio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::partial_token_ratio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreatePartialTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedPartialTokenRatio>(context, str);
        }
    };
}

static inline double WRatio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
static inline double WRatio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::WRatio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateWRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedWRatio>(context, str);
        }
    };
}

static inline double QRatio_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
static inline double QRatio_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return fuzz::QRatio(str1, str2, score_cutoff);
    });
}
static inline RfSimilarityFunctionTable CreateQRatioFunctionTable()
{
    return {
        nullptr,
        [](RfSimilarityContext* context, const RfKwargsContext*, const RfString* str) {
            return similarity_init<fuzz::CachedQRatio>(context, str);
        }
    };
}

/* string_metric */
static inline PyObject* levenshtein_no_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}
static inline PyObject* levenshtein_default_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::levenshtein(str1, str2, {insertion, deletion, substitution}, max);
    });
    return dist_to_long(result);
}

static inline double normalized_levenshtein_no_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}
static inline double normalized_levenshtein_default_process(const RfString& s1, const RfString& s2,
    size_t insertion, size_t deletion, size_t substitution, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_levenshtein(str1, str2, {insertion, deletion, substitution}, score_cutoff);
    });
}

static inline PyObject* hamming_no_process(const RfString& s1, const RfString& s2, size_t max)
{
    size_t result = visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}
static inline PyObject* hamming_default_process(const RfString& s1, const RfString& s2, size_t max)
{
    size_t result = visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::hamming(str1, str2, max);
    });
    return dist_to_long(result);
}

static inline double normalized_hamming_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}
static inline double normalized_hamming_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::normalized_hamming(str1, str2, score_cutoff);
    });
}

static inline double jaro_similarity_no_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}
static inline double jaro_similarity_default_process(const RfString& s1, const RfString& s2, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_similarity(str1, str2, score_cutoff);
    });
}

static inline double jaro_winkler_similarity_no_process(const RfString& s1, const RfString& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}
static inline double jaro_winkler_similarity_default_process(const RfString& s1, const RfString& s2,
    double prefix_weight, double score_cutoff)
{
    return visitor_default_process(s1, s2, [&](auto str1, auto str2) {
        return string_metric::jaro_winkler_similarity(str1, str2, prefix_weight, score_cutoff);
    });
}

static inline std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_no_process(
    const RfString& s1, const RfString& s2)
{
    return visitor(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });
}

static inline std::vector<rapidfuzz::LevenshteinEditOp> levenshtein_editops_default_process(
    const RfString& s1, const RfString& s2)
{
    return visitor_default_process(s1, s2, [](auto str1, auto str2) {
        return string_metric::levenshtein_editops(str1, str2);
    });     
}
