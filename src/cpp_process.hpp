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

struct RF_SimilarityWrapper {
    RF_Similarity context;

    RF_SimilarityWrapper()
      : context({nullptr, nullptr, nullptr}) {}
    explicit RF_SimilarityWrapper(RF_Similarity context_)
      : context(context_) {}

    RF_SimilarityWrapper(const RF_SimilarityWrapper&) = delete;
    RF_SimilarityWrapper& operator=(const RF_SimilarityWrapper&) = delete;

    RF_SimilarityWrapper(RF_SimilarityWrapper&& other)
     : context(other.context)
    {
        other.context = {nullptr, nullptr, nullptr};
    }

    RF_SimilarityWrapper& operator=(RF_SimilarityWrapper&& other) {
        if (&other != this) {
            if (context.deinit) {
                context.deinit(&context);
            }

            context = other.context;
            other.context = {nullptr, nullptr, nullptr};
      }
      return *this;
    };

    ~RF_SimilarityWrapper() {
        if (context.deinit) {
            context.deinit(&context);
        }
    }

    double similarity(const RF_String* str, double score_cutoff) {
        double sim;
        PyErr2RuntimeExn(context.similarity(&sim, &context, str, score_cutoff));
        return sim;
    }
};

struct RF_DistanceWrapper {
    RF_Distance context;

    RF_DistanceWrapper()
      : context({nullptr, nullptr, nullptr}) {}
    explicit RF_DistanceWrapper(RF_Distance context_)
      : context(context_) {}

    RF_DistanceWrapper(const RF_DistanceWrapper&) = delete;
    RF_DistanceWrapper& operator=(const RF_DistanceWrapper&) = delete;

    RF_DistanceWrapper(RF_DistanceWrapper&& other)
     : context(other.context)
    {
        other.context = {nullptr, nullptr, nullptr};
    }

    RF_DistanceWrapper& operator=(RF_DistanceWrapper&& other) {
        if (&other != this) {
            if (context.deinit) {
                context.deinit(&context);
            }

            context = other.context;
            other.context = {nullptr, nullptr, nullptr};
      }
      return *this;
    };

    ~RF_DistanceWrapper() {
        if (context.deinit) {
            context.deinit(&context);
        }
    }

    size_t distance(const RF_String* str, size_t max) {
        size_t dist;
        PyErr2RuntimeExn(context.distance(&dist, &context, str, max));
        return dist;
    }
};
