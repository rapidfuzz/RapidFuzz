#include "cpp_common.hpp"

#define RATIO_SINGLE(RATIO, RATIO_FUNC)                                            \
template<typename CharT>                                                           \
inline double RATIO##_single(proc_string s1, proc_string s2, double score_cutoff)  \
{                                                                                  \
    switch(s2.kind){                                                               \
    case PyUnicode_1BYTE_KIND:                                                     \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length),   \
            score_cutoff                                                           \
        );                                                                         \
    case PyUnicode_2BYTE_KIND:                                                     \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length), \
            score_cutoff                                                           \
        );                                                                         \
    default:                                                                       \
        return RATIO_FUNC(                                                         \
            rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length),       \
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length), \
            score_cutoff                                                           \
        );                                                                         \
    }                                                                              \
}

#define RATIO_IMPL(RATIO, RATIO_FUNC)                                \
double RATIO##_impl(PyObject* s1, PyObject* s2, double score_cutoff) \
{                                                                    \
    proc_string c_s1 = convert_string(s1, "s1 must be a String");    \
    proc_string c_s2 = convert_string(s2, "s2 must be a String");    \
                                                                     \
    switch(c_s1.kind){                                               \
    case PyUnicode_1BYTE_KIND:                                       \
        return RATIO##_single<uint8_t>(c_s1, c_s2, score_cutoff);    \
    case PyUnicode_2BYTE_KIND:                                       \
        return RATIO##_single<uint16_t>(c_s1, c_s2, score_cutoff);   \
    default:                                                         \
        return RATIO##_single<uint32_t>(c_s1, c_s2, score_cutoff);   \
    }                                                                \
}

#define RATIO_SINGLE_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                               \
template<typename CharT>                                                              \
inline double RATIO##_single_default_process(                                         \
    proc_string s1, proc_string s2, double score_cutoff)                              \
{                                                                                     \
    switch(s2.kind){                                                                  \
    case PyUnicode_1BYTE_KIND:                                                        \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)s2.data, s2.length)   \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    case PyUnicode_2BYTE_KIND:                                                        \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)s2.data, s2.length) \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    default:                                                                          \
        return RATIO_FUNC(                                                            \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<CharT>((CharT*)s1.data, s1.length)       \
            ),                                                                        \
            utils::default_process(                                                   \
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)s2.data, s2.length) \
            ),                                                                        \
            score_cutoff                                                              \
        );                                                                            \
    }                                                                                 \
}

#define RATIO_IMPL_DEFAULT_PROCESS(RATIO, RATIO_FUNC)                                   \
double RATIO##_impl_default_process(PyObject* s1, PyObject* s2, double score_cutoff) {  \
    proc_string c_s1 = convert_string(s1, "s1 must be a String");                       \
    proc_string c_s2 = convert_string(s2, "s2 must be a String");                       \
                                                                                        \
    switch(c_s1.kind){                                                                  \
    case PyUnicode_1BYTE_KIND:                                                          \
        return RATIO##_single_default_process<uint8_t>(c_s1, c_s2, score_cutoff);       \
    case PyUnicode_2BYTE_KIND:                                                          \
        return RATIO##_single_default_process<uint16_t>(c_s1, c_s2, score_cutoff);      \
    default:                                                                            \
        return RATIO##_single_default_process<uint32_t>(c_s1, c_s2, score_cutoff);      \
    }                                                                                   \
}

#define RATIO_DEF(RATIO, RATIO_FUNC)            \
RATIO_SINGLE_DEFAULT_PROCESS(RATIO, RATIO_FUNC) \
RATIO_IMPL_DEFAULT_PROCESS(RATIO, RATIO_FUNC)   \
RATIO_SINGLE(RATIO, RATIO_FUNC)                 \
RATIO_IMPL(RATIO, RATIO_FUNC)

RATIO_DEF(ratio,                    fuzz::ratio)
RATIO_DEF(partial_ratio,            fuzz::partial_ratio)
RATIO_DEF(token_sort_ratio,         fuzz::token_sort_ratio)
RATIO_DEF(token_set_ratio,          fuzz::token_set_ratio)
RATIO_DEF(token_ratio,              fuzz::token_ratio)
RATIO_DEF(partial_token_sort_ratio, fuzz::partial_token_sort_ratio)
RATIO_DEF(partial_token_set_ratio,  fuzz::partial_token_set_ratio)
RATIO_DEF(partial_token_ratio,      fuzz::partial_token_ratio)
RATIO_DEF(WRatio,                   fuzz::WRatio)
RATIO_DEF(QRatio,                   fuzz::QRatio)