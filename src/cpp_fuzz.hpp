#pragma once
#include "cpp_common.hpp"

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
