#pragma once
#include "cpp_common.hpp"

static inline double ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool RatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedRatio, double>(self, str_count, str);
}

static inline double partial_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::partial_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool PartialRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedPartialRatio, double>(self, str_count, str);
}


static inline double token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::token_sort_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool TokenSortRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedTokenSortRatio, double>(self, str_count, str);
}


static inline double token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::token_set_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool TokenSetRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedTokenSetRatio, double>(self, str_count, str);
}

static inline double token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::token_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool TokenRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedTokenRatio, double>(self, str_count, str);
}

static inline double partial_token_sort_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::partial_token_sort_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool PartialTokenSortRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedPartialTokenSortRatio, double>(self, str_count, str);
}

static inline double partial_token_set_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::partial_token_set_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool PartialTokenSetRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedPartialTokenSetRatio, double>(self, str_count, str);
}

static inline double partial_token_ratio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::partial_token_ratio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool PartialTokenRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedPartialTokenRatio, double>(self, str_count, str);
}

static inline double WRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::WRatio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool WRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedWRatio, double>(self, str_count, str);
}

static inline double QRatio_func(const RF_String& s1, const RF_String& s2, double score_cutoff)
{
    return visitor(s1, s2, [&](auto first1, auto last1, auto first2, auto last2) {
        return fuzz::QRatio(first1, last1, first2, last2, score_cutoff);
    });
}
static inline bool QRatioInit(RF_ScorerFunc* self, const RF_Kwargs*, int64_t str_count, const RF_String* str)
{
    return similarity_init<fuzz::CachedQRatio, double>(self, str_count, str);
}
