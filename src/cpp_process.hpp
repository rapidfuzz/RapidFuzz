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

struct CachedScorerContext {
    RfSimilarityContext context;

    CachedScorerContext()
      : context({nullptr, nullptr, nullptr}) {}
    explicit CachedScorerContext(RfSimilarityContext context_)
      : context(context_) {}

    CachedScorerContext(const CachedScorerContext&) = delete;
    CachedScorerContext& operator=(const CachedScorerContext&) = delete;

    CachedScorerContext(CachedScorerContext&& other)
     : context(other.context)
    {
        other.context = {nullptr, nullptr, nullptr};
    }

    CachedScorerContext& operator=(CachedScorerContext&& other) {
        if (&other != this) {
            if (context.deinit) {
                context.deinit(&context);
            }

            context = other.context;
            other.context = {nullptr, nullptr, nullptr};
      }
      return *this;
    };

    ~CachedScorerContext() {
        if (context.deinit) {
            context.deinit(&context);
        }
    }

    double similarity(const RfString* str, double score_cutoff) {
        double sim;
        PyErr2RuntimeExn(context.similarity(&sim, &context, str, score_cutoff));
        return sim;
    }
};

struct CachedDistanceContext {
    RfDistanceContext context;

    CachedDistanceContext()
      : context({nullptr, nullptr, nullptr}) {}
    explicit CachedDistanceContext(RfDistanceContext context_)
      : context(context_) {}

    CachedDistanceContext(const CachedDistanceContext&) = delete;
    CachedDistanceContext& operator=(const CachedDistanceContext&) = delete;

    CachedDistanceContext(CachedDistanceContext&& other)
     : context(other.context)
    {
        other.context = {nullptr, nullptr, nullptr};
    }

    CachedDistanceContext& operator=(CachedDistanceContext&& other) {
        if (&other != this) {
            if (context.deinit) {
                context.deinit(&context);
            }

            context = other.context;
            other.context = {nullptr, nullptr, nullptr};
      }
      return *this;
    };

    ~CachedDistanceContext() {
        if (context.deinit) {
            context.deinit(&context);
        }
    }

    size_t distance(const RfString* str, size_t max) {
        size_t dist;
        PyErr2RuntimeExn(context.distance(&dist, &context, str, max));
        return dist;
    }
};
