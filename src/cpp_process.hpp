#include "cpp_common.hpp"

struct ListMatchElem {
    double score;
    size_t index;
};

struct DictMatchElem {
    double score;
    size_t index;
    PyObject* choice;
    PyObject* key;
};

struct ExtractComp
{
    template<class T>
    bool operator()(T const &a, T const &b) const {
        if (a.score > b.score) {
            return true;
        } else if (a.score < b.score) {
            return false;
        } else {
            return a.index < b.index;
        }
    }
};

typedef double (*scorer_func) (void* context, PyObject* str, double score_cutoff);
typedef void (*scorer_context_deinit) (void* context);

struct scorer_context {
    void* context;
    scorer_func scorer;
    scorer_context_deinit deinit;
};

template <typename CachedScorer>
static void cached_deinit(void* context)
{
    delete (CachedScorer*)context;
}

template<typename CachedScorer>
static inline double cached_func_default_process(
    void* context, PyObject* py_str, double score_cutoff)
{
    proc_string str = convert_string(py_str, "choice must be a String or None");
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
    case PyUnicode_1BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length)),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length)),
            score_cutoff
        );
    case PyUnicode_4BYTE_KIND:
        return ratio->ratio(
            utils::default_process(
                rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length)),
            score_cutoff
        );
    default:
       throw std::logic_error("Reached end of control flow in cached_func_default_process");
    }
}

template<typename CachedScorer>
static inline double cached_func(void* context, PyObject* py_str, double score_cutoff)
{
    proc_string str = convert_string(py_str, "choice must be a String or None");
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
    case PyUnicode_1BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint8_t>((uint8_t*)str.data, str.length),
            score_cutoff
        );
    case PyUnicode_2BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint16_t>((uint16_t*)str.data, str.length),
            score_cutoff
        );
    case PyUnicode_4BYTE_KIND:
        return ratio->ratio(
            rapidfuzz::basic_string_view<uint32_t>((uint32_t*)str.data, str.length),
            score_cutoff
        );
    default:
       throw std::logic_error("Reached end of control flow in cached_func");
    }
}

template<template <typename> class CachedScorer, typename CharT, typename ...Args>
static inline scorer_context get_scorer_context(const proc_string& str, int def_process, Args... args)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    scorer_context context;
    context.context = (void*) new CachedScorer<Sentence>(Sentence((CharT*)str.data, str.length), args...);

    if (def_process) {
        context.scorer = cached_func_default_process<CachedScorer<Sentence>>;
    } else {
        context.scorer = cached_func<CachedScorer<Sentence>>;
    }
    context.deinit = cached_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline scorer_context cached_init(PyObject* py_str, int def_process, Args... args)
{
    proc_string str = convert_string(py_str, "query must be a String");

    switch(str.kind){
    case PyUnicode_1BYTE_KIND:
        return get_scorer_context<CachedScorer, uint8_t>(str, def_process, args...);
    case PyUnicode_2BYTE_KIND:
        return get_scorer_context<CachedScorer, uint16_t>(str, def_process, args...);
    case PyUnicode_4BYTE_KIND:
        return get_scorer_context<CachedScorer, uint32_t>(str, def_process, args...);
    default:
       throw std::logic_error("Reached end of control flow in cached_init");
    }
}

/* fuzz */
static scorer_context cached_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedRatio>(py_str, def_process);
}

static scorer_context cached_partial_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialRatio>(py_str, def_process);
}

static scorer_context cached_token_sort_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenSortRatio>(py_str, def_process);
}

static scorer_context cached_token_set_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenSetRatio>(py_str, def_process);
}

static scorer_context cached_token_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedTokenRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_sort_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenSortRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_set_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenSetRatio>(py_str, def_process);
}

static scorer_context cached_partial_token_ratio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedPartialTokenRatio>(py_str, def_process);
}

static scorer_context cached_WRatio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedWRatio>(py_str, def_process);
}

static scorer_context cached_QRatio_init(PyObject* py_str, int def_process)
{
    return cached_init<fuzz::CachedQRatio>(py_str, def_process);
}

/* string_metric */

static scorer_context cached_normalized_levenshtein_init(PyObject* py_str, int def_process,
  size_t insertion, size_t deletion, size_t substitution)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};
    return cached_init<string_metric::CachedNormalizedLevenshtein>(
        py_str, def_process, weights);
}

static scorer_context cached_normalized_hamming_init(PyObject* py_str, int def_process)
{
    return cached_init<string_metric::CachedNormalizedHamming>(py_str, def_process);
}
