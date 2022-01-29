#pragma once
#include "cpp_common.hpp"

template <typename T>
struct ListMatchElem
{
    ListMatchElem() {}
    ListMatchElem(T score, int64_t index, PyObjectWrapper choice)
        : score(score), index(index), choice(std::move(choice)) {}

    T score;
    int64_t index;
    PyObjectWrapper choice;
};

template <typename T>
struct DictMatchElem
{
    DictMatchElem() {}
    DictMatchElem(T score, int64_t index, PyObjectWrapper choice, PyObjectWrapper key)
        : score(score), index(index), choice(std::move(choice)), key(std::move(key)) {}

    T score;
    int64_t index;
    PyObjectWrapper choice;
    PyObjectWrapper key;
};

struct DictStringElem
{
    DictStringElem() : index(-1) {}
    DictStringElem(int64_t index, PyObjectWrapper key, PyObjectWrapper val, RF_StringWrapper proc_val)
        : index(index), key(std::move(key)), val(std::move(val)), proc_val(std::move(proc_val)) {}

    int64_t index;
    PyObjectWrapper key;
    PyObjectWrapper val;
    RF_StringWrapper proc_val;
};

struct ListStringElem
{
    ListStringElem() : index(-1) {}
    ListStringElem(int64_t index, PyObjectWrapper val, RF_StringWrapper proc_val)
        : index(index), val(std::move(val)), proc_val(std::move(proc_val)) {}

    int64_t index;
    PyObjectWrapper val;
    RF_StringWrapper proc_val;
};

struct ExtractComp
{
    ExtractComp()
        : m_scorer_flags(nullptr) { }

    explicit ExtractComp(const RF_ScorerFlags* scorer_flags)
        : m_scorer_flags(scorer_flags) { }

    template<typename T>
    bool operator()(T const &a, T const &b) const {
        if (m_scorer_flags->flags & RF_SCORER_FLAG_RESULT_F64)
        {
            return is_first(a, b, m_scorer_flags->optimal_score.f64, m_scorer_flags->worst_score.f64);
        }
        else
        {
            return is_first(a, b, m_scorer_flags->optimal_score.i64, m_scorer_flags->worst_score.i64);
        }
    }

private:
    template<typename T, typename U>
    static bool is_first(T const &a, T const &b, U optimal, U worst) {
        if (optimal > worst)
        {
            if (a.score > b.score) {
                return true;
            } else if (a.score < b.score) {
                return false;
            }
        }
        else
        {
            if (a.score > b.score) {
                return false;
            } else if (a.score < b.score) {
                return true;
            }
        }
        return a.index < b.index;
    }

    const RF_ScorerFlags* m_scorer_flags;
};

struct RF_ScorerWrapper {
    RF_ScorerFunc scorer_func;

    RF_ScorerWrapper()
      : scorer_func({nullptr, nullptr, nullptr}) {}
    explicit RF_ScorerWrapper(RF_ScorerFunc scorer_func_)
      : scorer_func(scorer_func_) {}

    RF_ScorerWrapper(const RF_ScorerWrapper&) = delete;
    RF_ScorerWrapper& operator=(const RF_ScorerWrapper&) = delete;

    RF_ScorerWrapper(RF_ScorerWrapper&& other)
     : scorer_func(other.scorer_func)
    {
        other.scorer_func = {nullptr, nullptr, nullptr};
    }

    RF_ScorerWrapper& operator=(RF_ScorerWrapper&& other) {
        if (&other != this) {
            if (scorer_func.dtor) {
                scorer_func.dtor(&scorer_func);
            }

            scorer_func = other.scorer_func;
            other.scorer_func = {nullptr, nullptr, nullptr};
      }
      return *this;
    };

    ~RF_ScorerWrapper() {
        if (scorer_func.dtor) {
            scorer_func.dtor(&scorer_func);
        }
    }

    void call(const RF_String* str, double score_cutoff, double* result) const {
        PyErr2RuntimeExn(scorer_func.call.f64(&scorer_func, str, 1, score_cutoff, result));
    }

    void call(const RF_String* str, int64_t score_cutoff, int64_t* result) const {
        PyErr2RuntimeExn(scorer_func.call.i64(&scorer_func, str, 1, score_cutoff, result));
    }
};

template <typename T>
bool is_lowest_score_worst(const RF_ScorerFlags* scorer_flags)
{
    if (std::is_same<T, double>::value)
    {
        return scorer_flags->optimal_score.f64 > scorer_flags->worst_score.f64;
    }
    else
    {
        return scorer_flags->optimal_score.i64 > scorer_flags->worst_score.i64;
    }
}

template <typename T>
T get_optimal_score(const RF_ScorerFlags* scorer_flags)
{
    if (std::is_same<T, double>::value)
    {
        return (T)scorer_flags->optimal_score.f64;
    }
    else
    {
        return (T)scorer_flags->optimal_score.i64;
    }
}

template <typename T>
std::vector<DictMatchElem<T>> extract_dict_impl(
    const RF_Kwargs* kwargs, const RF_ScorerFlags* scorer_flags, RF_Scorer* scorer,
    const RF_StringWrapper& query, const std::vector<DictStringElem>& choices, T score_cutoff)
{
    std::vector<DictMatchElem<T>> results;
    results.reserve(choices.size());

    RF_ScorerFunc scorer_func;
    PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &query.string));
    RF_ScorerWrapper ScorerFunc(scorer_func);

    bool lowest_score_worst = is_lowest_score_worst<T>(scorer_flags);

    for (const auto& choice : choices)
    {
        T score;
        ScorerFunc.call(&choice.proc_val.string, score_cutoff, &score);

        if (lowest_score_worst)
        {
            if (score >= score_cutoff)
            {
                results.emplace_back(score, choice.index, choice.val, choice.key);
            }
        }
        else
        {
            if (score <= score_cutoff)
            {
                results.emplace_back(score, choice.index, choice.val, choice.key);
            }
        }
    }

    return results;
}


template <typename T>
std::vector<ListMatchElem<T>> extract_list_impl(
    const RF_Kwargs* kwargs, const RF_ScorerFlags* scorer_flags, RF_Scorer* scorer,
    const RF_StringWrapper& query, const std::vector<ListStringElem>& choices, T score_cutoff)
{
    std::vector<ListMatchElem<T>> results;
    results.reserve(choices.size());

    RF_ScorerFunc scorer_func;
    PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &query.string));
    RF_ScorerWrapper ScorerFunc(scorer_func);

    bool lowest_score_worst = is_lowest_score_worst<T>(scorer_flags);

    for (const auto& choice : choices)
    {
        T score;
        ScorerFunc.call(&choice.proc_val.string, score_cutoff, &score);

        if (lowest_score_worst)
        {
            if (score >= score_cutoff)
            {
                results.emplace_back(score, choice.index, choice.val);
            }
        }
        else
        {
            if (score <= score_cutoff)
            {
                results.emplace_back(score, choice.index, choice.val);
            }
        }
    }

    return results;
}
