#include "cpp_common.hpp"

struct DictElem {
    PyObject* key;
    PyObject* value;
};

struct ListMatchScorerElem {
    double score;
    size_t index;
    PyObject* choice;
};

struct DictMatchScorerElem {
    double score;
    size_t index;
    PyObject* choice;
    PyObject* key;
};

struct ListMatchDistanceElem {
    std::size_t distance;
    size_t index;
    PyObject* choice;
};

struct DictMatchDistanceElem {
    std::size_t distance;
    size_t index;
    PyObject* choice;
    PyObject* key;
};


struct ExtractScorerComp
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

struct ExtractDistanceComp
{
    template<class T>
    bool operator()(T const &a, T const &b) const {
        if (a.distance < b.distance) {
            return true;
        } else if (a.distance > b.distance) {
            return false;
        } else {
            return a.index < b.index;
        }
    }
};

typedef double (*scorer_func) (void* context, const proc_string& str, double score_cutoff);
typedef std::size_t (*distance_func) (void* context, const proc_string& str, std::size_t max);
typedef void (*context_deinit) (void* context);

struct CachedScorerContext {
    void* context;
    scorer_func scorer;
    context_deinit deinit;

    CachedScorerContext()
      : context(nullptr), scorer(nullptr), deinit(nullptr) {}
    CachedScorerContext(void* _context, scorer_func _scorer, context_deinit _deinit)
      : context(_context), scorer(_scorer), deinit(_deinit) {}

    CachedScorerContext(const CachedScorerContext&) = delete;
    CachedScorerContext& operator=(const CachedScorerContext&) = delete;

    CachedScorerContext(CachedScorerContext&& other)
     : context(other.context), scorer(other.scorer), deinit(other.deinit)
    {
        other.context = nullptr;
    }

    CachedScorerContext& operator=(CachedScorerContext&& other) {
        if (&other != this) {
            if (deinit && context) {
                deinit(context);
            }

            context = other.context;
            scorer = other.scorer;
            deinit = other.deinit;

            other.context = nullptr;
      }
      return *this;
    };

    ~CachedScorerContext() {
        if (deinit && context) {
            deinit(context);
        }  
    }

    double ratio(const proc_string& str, double score_cutoff) {
        return scorer(context, str, score_cutoff);
    }
};

struct CachedDistanceContext {
    void* context;
    distance_func scorer;
    context_deinit deinit;

    CachedDistanceContext()
      : context(nullptr), scorer(nullptr), deinit(nullptr) {}
    CachedDistanceContext(void* _context, distance_func _scorer, context_deinit _deinit)
      : context(_context), scorer(_scorer), deinit(_deinit) {}

    CachedDistanceContext(const CachedDistanceContext&) = delete;
    CachedDistanceContext& operator=(const CachedDistanceContext&) = delete;

    CachedDistanceContext(CachedDistanceContext&& other)
     : context(other.context), scorer(other.scorer), deinit(other.deinit)
    {
        other.context = nullptr;
    }

    CachedDistanceContext& operator=(CachedDistanceContext&& other) {
        if (&other != this) {
            if (deinit && context) {
                deinit(context);
            }

            context = other.context;
            scorer = other.scorer;
            deinit = other.deinit;

            other.context = nullptr;
      }
      return *this;
    };

    ~CachedDistanceContext() {
        if (deinit && context) {
            deinit(context);
        }  
    }

    std::size_t ratio(const proc_string& str, std::size_t max) {
        return scorer(context, str, max);
    }
};

template <typename CachedScorer>
static void cached_deinit(void* context)
{
    delete (CachedScorer*)context;
}

template<typename CachedScorer>
static inline double cached_scorer_func_default_process(
    void* context, const proc_string& str, double score_cutoff)
{
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return ratio->ratio(default_process<TYPE>(str), score_cutoff);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_scorer_func_default_process");
    }
}

template<typename CachedScorer>
static inline double cached_scorer_func(void* context, const proc_string& str, double score_cutoff)
{
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return ratio->ratio(no_process<TYPE>(str), score_cutoff);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_scorer_func");
    }
}

template<template <typename> class CachedScorer, typename CharT, typename ...Args>
static inline CachedScorerContext get_CachedScorerContext(const proc_string& str, int def_process, Args... args)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    CachedScorerContext context;
    context.context = (void*) new CachedScorer<Sentence>(Sentence((CharT*)str.data, str.length), args...);

    if (def_process) {
        context.scorer = cached_scorer_func_default_process<CachedScorer<Sentence>>;
    } else {
        context.scorer = cached_scorer_func<CachedScorer<Sentence>>;
    }
    context.deinit = cached_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline CachedScorerContext cached_scorer_init(const proc_string& str, int def_process, Args... args)
{
    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return get_CachedScorerContext<CachedScorer, TYPE>(str, def_process, args...);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_scorer_init");
    }
}

/* fuzz */
static CachedScorerContext cached_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedRatio>(str, def_process);
}

static CachedScorerContext cached_partial_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedPartialRatio>(str, def_process);
}

static CachedScorerContext cached_token_sort_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedTokenSortRatio>(str, def_process);
}

static CachedScorerContext cached_token_set_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedTokenSetRatio>(str, def_process);
}

static CachedScorerContext cached_token_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedTokenRatio>(str, def_process);
}

static CachedScorerContext cached_partial_token_sort_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedPartialTokenSortRatio>(str, def_process);
}

static CachedScorerContext cached_partial_token_set_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedPartialTokenSetRatio>(str, def_process);
}

static CachedScorerContext cached_partial_token_ratio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedPartialTokenRatio>(str, def_process);
}

static CachedScorerContext cached_WRatio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedWRatio>(str, def_process);
}

static CachedScorerContext cached_QRatio_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<fuzz::CachedQRatio>(str, def_process);
}

/* string_metric */

static CachedScorerContext cached_normalized_levenshtein_init(const proc_string& str, int def_process,
  size_t insertion, size_t deletion, size_t substitution)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};
    return cached_scorer_init<string_metric::CachedNormalizedLevenshtein>(
        str, def_process, weights);
}

static CachedScorerContext cached_normalized_hamming_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<string_metric::CachedNormalizedHamming>(str, def_process);
}

static CachedScorerContext cached_jaro_winkler_similarity_init(const proc_string& str, int def_process, double prefix_weight)
{
    return cached_scorer_init<string_metric::CachedJaroWinklerSimilarity>(str, def_process, prefix_weight);
}

static CachedScorerContext cached_jaro_similarity_init(const proc_string& str, int def_process)
{
    return cached_scorer_init<string_metric::CachedJaroSimilarity>(str, def_process);
}


/*************************************************
 *               cached distances
 *************************************************/

template<typename CachedDistance>
static inline std::size_t cached_distance_func_default_process(
    void* context, const proc_string& str, std::size_t max)
{
    CachedDistance* distance = (CachedDistance*)context;

    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return distance->distance(default_process<TYPE>(str), max);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_distance_func_default_process");
    }
}

template<typename CachedDistance>
static inline std::size_t cached_distance_func(void* context, const proc_string& str, std::size_t max)
{
    CachedDistance* distance = (CachedDistance*)context;

    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return distance->distance(no_process<TYPE>(str), max);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_distance_func");
    }
}

template<template <typename> class CachedDistance, typename CharT, typename ...Args>
static inline CachedDistanceContext get_CachedDistanceContext(const proc_string& str, int def_process, Args... args)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    CachedDistanceContext context;
    context.context = (void*) new CachedDistance<Sentence>(Sentence((CharT*)str.data, str.length), args...);

    if (def_process) {
        context.scorer = cached_distance_func_default_process<CachedDistance<Sentence>>;
    } else {
        context.scorer = cached_distance_func<CachedDistance<Sentence>>;
    }
    context.deinit = cached_deinit<CachedDistance<Sentence>>;
    return context;
}

template<template <typename> class CachedDistance, typename ...Args>
static inline CachedDistanceContext cached_distance_init(const proc_string& str, int def_process, Args... args)
{
    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return get_CachedDistanceContext<CachedDistance, TYPE>(str, def_process, args...);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in cached_distance_init");
    }
}

/* string_metric */

static CachedDistanceContext cached_levenshtein_init(const proc_string& str, int def_process,
  size_t insertion, size_t deletion, size_t substitution)
{
    rapidfuzz::LevenshteinWeightTable weights = {insertion, deletion, substitution};
    return cached_distance_init<string_metric::CachedLevenshtein>(
        str, def_process, weights);
}

static CachedDistanceContext cached_hamming_init(const proc_string& str, int def_process)
{
    return cached_distance_init<string_metric::CachedHamming>(str, def_process);
}
