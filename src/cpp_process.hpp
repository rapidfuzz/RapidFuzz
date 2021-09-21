#pragma once
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

struct KwargsContext {
    void* context;
    context_deinit deinit;

    KwargsContext()
      : context(nullptr), deinit(nullptr) {}
    KwargsContext(void* _context, context_deinit _deinit)
      : context(_context), deinit(_deinit) {}

    KwargsContext(const KwargsContext&) = delete;
    KwargsContext& operator=(const KwargsContext&) = delete;

    KwargsContext(KwargsContext&& other)
      : context(other.context), deinit(other.deinit)
    {
        other.context = nullptr;
    }

    KwargsContext& operator=(KwargsContext&& other) {
        if (&other != this) {
            if (deinit && context) {
                deinit(context);
            }

            context = other.context;
            deinit = other.deinit;

            other.context = nullptr;
        }
        return *this;
    };

    ~KwargsContext() {
        if (deinit && context) {
            deinit(context);
        }  
    }
};

template <typename CachedScorer>
static void cached_deinit(void* context)
{
    delete (CachedScorer*)context;
}

typedef KwargsContext (*kwargs_context_init)(PyObject* kwds);
typedef CachedDistanceContext (*distance_context_init)(const KwargsContext& kwargs, const proc_string& str);
typedef CachedScorerContext (*scorer_context_init)(const KwargsContext& kwargs, const proc_string& str);

struct ScorerFunctionTable {
    kwargs_context_init kwargs_init;
    scorer_context_init init;
};

struct DistanceFunctionTable {
    kwargs_context_init kwargs_init;
    distance_context_init init;
};

template<typename CachedScorer>
static inline double scorer_func_wrapper(void* context, const proc_string& str, double score_cutoff)
{
    CachedScorer* ratio = (CachedScorer*)context;

    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return ratio->ratio(no_process<TYPE>(str), score_cutoff);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in scorer_func");
    }
}

template<template <typename> class CachedScorer, typename CharT, typename ...Args>
static inline CachedScorerContext get_ScorerContext(const proc_string& str, Args... args)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    CachedScorerContext context;
    context.context = (void*) new CachedScorer<Sentence>(Sentence((CharT*)str.data, str.length), args...);

    context.scorer = scorer_func_wrapper<CachedScorer<Sentence>>;
    context.deinit = cached_deinit<CachedScorer<Sentence>>;
    return context;
}

template<template <typename> class CachedScorer, typename ...Args>
static inline CachedScorerContext scorer_init(const proc_string& str, Args... args)
{
    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return get_ScorerContext<CachedScorer, TYPE>(str, args...);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in scorer_init");
    }
}

/* fuzz */
static ScorerFunctionTable CreateRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedRatio>(str);
        }
    };
}

static ScorerFunctionTable CreatePartialRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedPartialRatio>(str);
        }
    };
}

static ScorerFunctionTable CreateTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedTokenSortRatio>(str);
        }
    };
}

static ScorerFunctionTable CreateTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedTokenSetRatio>(str);
        }
    };
}

static ScorerFunctionTable CreateTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedTokenRatio>(str);
        }
    };
}

static ScorerFunctionTable CreatePartialTokenSortRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedPartialTokenSortRatio>(str);
        }
    };
}

static ScorerFunctionTable CreatePartialTokenSetRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedPartialTokenSetRatio>(str);
        }
    };
}

static ScorerFunctionTable CreatePartialTokenRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedPartialTokenRatio>(str);
        }
    };
}

static ScorerFunctionTable CreateWRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedWRatio>(str);
        }
    };
}

static ScorerFunctionTable CreateQRatioFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<fuzz::CachedQRatio>(str);
        }
    };
}

/* string_metric */
static CachedScorerContext cached_normalized_levenshtein_init(const KwargsContext& kwargs, const proc_string& str)
{
    return scorer_init<string_metric::CachedNormalizedLevenshtein>(
        str, *static_cast<rapidfuzz::LevenshteinWeightTable*>(kwargs.context));
}

static ScorerFunctionTable CreateNormalizedHammingFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<string_metric::CachedNormalizedHamming>(str);
        }
    };
}

static CachedScorerContext cached_jaro_winkler_similarity_init(const KwargsContext& kwargs, const proc_string& str)
{
    return scorer_init<string_metric::CachedJaroWinklerSimilarity>(
        str, *static_cast<double*>(kwargs.context));
}

static ScorerFunctionTable CreateJaroSimilarityFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return scorer_init<string_metric::CachedJaroSimilarity>(str);
        }
    };
}

/*************************************************
 *               cached distances
 *************************************************/

template<typename CachedDistance>
static inline std::size_t distance_func_wrapper(void* context, const proc_string& str, std::size_t max)
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
static inline CachedDistanceContext get_DistanceContext(const proc_string& str, Args... args)
{
    using Sentence = rapidfuzz::basic_string_view<CharT>;
    CachedDistanceContext context;
    context.context = (void*) new CachedDistance<Sentence>(Sentence((CharT*)str.data, str.length), args...);
    context.scorer = distance_func_wrapper<CachedDistance<Sentence>>;
    context.deinit = cached_deinit<CachedDistance<Sentence>>;
    return context;
}

template<template <typename> class CachedDistance, typename ...Args>
static inline CachedDistanceContext distance_init(const proc_string& str, Args... args)
{
    switch(str.kind){
# define X_ENUM(KIND, TYPE, ...) case KIND: return get_DistanceContext<CachedDistance, TYPE>(str, args...);
        LIST_OF_CASES()
# undef X_ENUM
    default:
       throw std::logic_error("Reached end of control flow in distance_init");
    }
}

static DistanceFunctionTable CreateHammingFunctionTable()
{
    return {
        nullptr,
        [](const KwargsContext&, const proc_string& str) {
            return distance_init<string_metric::CachedHamming>(str);
        }
    };
}

static CachedDistanceContext cached_levenshtein_init(const KwargsContext& kwargs, const proc_string& str)
{
    return distance_init<string_metric::CachedLevenshtein>(str, *static_cast<rapidfuzz::LevenshteinWeightTable*>(kwargs.context));
}
