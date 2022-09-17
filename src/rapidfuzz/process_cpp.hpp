#pragma once
#include "cpp_common.hpp"
#include "rapidfuzz_capi.h"
#include "taskflow/taskflow.hpp"
#include <atomic>
#include <chrono>
#include <exception>
using namespace std::chrono_literals;

template <typename T>
struct ListMatchElem {
    ListMatchElem()
    {}
    ListMatchElem(T score, int64_t index, PyObjectWrapper choice)
        : score(score), index(index), choice(std::move(choice))
    {}

    T score;
    int64_t index;
    PyObjectWrapper choice;
};

template <typename T>
struct DictMatchElem {
    DictMatchElem()
    {}
    DictMatchElem(T score, int64_t index, PyObjectWrapper choice, PyObjectWrapper key)
        : score(score), index(index), choice(std::move(choice)), key(std::move(key))
    {}

    T score;
    int64_t index;
    PyObjectWrapper choice;
    PyObjectWrapper key;
};

struct DictStringElem {
    DictStringElem() : index(-1)
    {}
    DictStringElem(int64_t index, PyObjectWrapper key, PyObjectWrapper val, RF_StringWrapper proc_val)
        : index(index), key(std::move(key)), val(std::move(val)), proc_val(std::move(proc_val))
    {}

    int64_t index;
    PyObjectWrapper key;
    PyObjectWrapper val;
    RF_StringWrapper proc_val;
};

struct ListStringElem {
    ListStringElem() : index(-1)
    {}
    ListStringElem(int64_t index, PyObjectWrapper val, RF_StringWrapper proc_val)
        : index(index), val(std::move(val)), proc_val(std::move(proc_val))
    {}

    int64_t index;
    PyObjectWrapper val;
    RF_StringWrapper proc_val;
};

struct ExtractComp {
    ExtractComp() : m_scorer_flags(nullptr)
    {}

    explicit ExtractComp(const RF_ScorerFlags* scorer_flags) : m_scorer_flags(scorer_flags)
    {}

    template <typename T>
    bool operator()(T const& a, T const& b) const
    {
        if (m_scorer_flags->flags & RF_SCORER_FLAG_RESULT_F64) {
            return is_first(a, b, m_scorer_flags->optimal_score.f64, m_scorer_flags->worst_score.f64);
        }
        else {
            return is_first(a, b, m_scorer_flags->optimal_score.i64, m_scorer_flags->worst_score.i64);
        }
    }

private:
    template <typename T, typename U>
    static bool is_first(T const& a, T const& b, U optimal, U worst)
    {
        if (optimal > worst) {
            if (a.score > b.score) {
                return true;
            }
            else if (a.score < b.score) {
                return false;
            }
        }
        else {
            if (a.score > b.score) {
                return false;
            }
            else if (a.score < b.score) {
                return true;
            }
        }
        return a.index < b.index;
    }

    const RF_ScorerFlags* m_scorer_flags;
};

struct RF_ScorerWrapper {
    RF_ScorerFunc scorer_func;

    RF_ScorerWrapper() : scorer_func({nullptr, {nullptr}, nullptr})
    {}
    explicit RF_ScorerWrapper(RF_ScorerFunc scorer_func_) : scorer_func(scorer_func_)
    {}

    RF_ScorerWrapper(const RF_ScorerWrapper&) = delete;
    RF_ScorerWrapper& operator=(const RF_ScorerWrapper&) = delete;

    RF_ScorerWrapper(RF_ScorerWrapper&& other) : scorer_func(other.scorer_func)
    {
        other.scorer_func = {nullptr, {nullptr}, nullptr};
    }

    RF_ScorerWrapper& operator=(RF_ScorerWrapper&& other)
    {
        if (&other != this) {
            if (scorer_func.dtor) {
                scorer_func.dtor(&scorer_func);
            }

            scorer_func = other.scorer_func;
            other.scorer_func = {nullptr, {nullptr}, nullptr};
        }
        return *this;
    };

    ~RF_ScorerWrapper()
    {
        if (scorer_func.dtor) {
            scorer_func.dtor(&scorer_func);
        }
    }

    void call(const RF_String* str, double score_cutoff, double* result) const
    {
        PyErr2RuntimeExn(scorer_func.call.f64(&scorer_func, str, 1, score_cutoff, result));
    }

    void call(const RF_String* str, int64_t score_cutoff, int64_t* result) const
    {
        PyErr2RuntimeExn(scorer_func.call.i64(&scorer_func, str, 1, score_cutoff, result));
    }
};

template <typename T>
bool is_lowest_score_worst(const RF_ScorerFlags* scorer_flags)
{
    if (std::is_same<T, double>::value) {
        return scorer_flags->optimal_score.f64 > scorer_flags->worst_score.f64;
    }
    else {
        return scorer_flags->optimal_score.i64 > scorer_flags->worst_score.i64;
    }
}

template <typename T>
T get_optimal_score(const RF_ScorerFlags* scorer_flags)
{
    if (std::is_same<T, double>::value) {
        return (T)scorer_flags->optimal_score.f64;
    }
    else {
        return (T)scorer_flags->optimal_score.i64;
    }
}

template <typename T>
std::vector<DictMatchElem<T>> extract_dict_impl(const RF_Kwargs* kwargs, const RF_ScorerFlags* scorer_flags,
                                                RF_Scorer* scorer, const RF_StringWrapper& query,
                                                const std::vector<DictStringElem>& choices, T score_cutoff)
{
    std::vector<DictMatchElem<T>> results;
    results.reserve(choices.size());

    RF_ScorerFunc scorer_func;
    PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &query.string));
    RF_ScorerWrapper ScorerFunc(scorer_func);

    bool lowest_score_worst = is_lowest_score_worst<T>(scorer_flags);

    for (size_t i = 0; i < choices.size(); ++i) {
        if (i % 1000 == 0)
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("");

        T score;
        ScorerFunc.call(&choices[i].proc_val.string, score_cutoff, &score);

        if (lowest_score_worst) {
            if (score >= score_cutoff) {
                results.emplace_back(score, choices[i].index, choices[i].val, choices[i].key);
            }
        }
        else {
            if (score <= score_cutoff) {
                results.emplace_back(score, choices[i].index, choices[i].val, choices[i].key);
            }
        }
    }

    return results;
}

template <typename T>
std::vector<ListMatchElem<T>> extract_list_impl(const RF_Kwargs* kwargs, const RF_ScorerFlags* scorer_flags,
                                                RF_Scorer* scorer, const RF_StringWrapper& query,
                                                const std::vector<ListStringElem>& choices, T score_cutoff)
{
    std::vector<ListMatchElem<T>> results;
    results.reserve(choices.size());

    RF_ScorerFunc scorer_func;
    PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &query.string));
    RF_ScorerWrapper ScorerFunc(scorer_func);

    bool lowest_score_worst = is_lowest_score_worst<T>(scorer_flags);

    for (size_t i = 0; i < choices.size(); ++i) {
        if (i % 1000 == 0)
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("");

        T score;
        ScorerFunc.call(&choices[i].proc_val.string, score_cutoff, &score);

        if (lowest_score_worst) {
            if (score >= score_cutoff) {
                results.emplace_back(score, choices[i].index, choices[i].val);
            }
        }
        else {
            if (score <= score_cutoff) {
                results.emplace_back(score, choices[i].index, choices[i].val);
            }
        }
    }

    return results;
}

int64_t any_round(double score)
{
    return std::llround(score);
}

int64_t any_round(int64_t score)
{
    return score;
}

enum class MatrixType {
    UNDEFINED,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
};

struct Matrix {
    MatrixType m_dtype;
    size_t m_rows;
    size_t m_cols;
    void* m_matrix;

    Matrix() : m_dtype(MatrixType::FLOAT32), m_rows(0), m_cols(0), m_matrix(nullptr)
    {}

    Matrix(MatrixType dtype, size_t rows, size_t cols) : m_dtype(dtype), m_rows(rows), m_cols(cols)
    {

        m_matrix = malloc(get_dtype_size() * m_rows * m_cols);
        if (m_matrix == nullptr) throw std::bad_alloc();
    }

    Matrix(const Matrix& other) : m_dtype(other.m_dtype), m_rows(other.m_rows), m_cols(other.m_cols)
    {
        m_matrix = malloc(get_dtype_size() * m_rows * m_cols);
        if (m_matrix == nullptr) throw std::bad_alloc();

        memcpy(m_matrix, other.m_matrix, get_dtype_size() * m_rows * m_cols);
    }

    Matrix(Matrix&& other) noexcept : m_dtype(MatrixType::FLOAT32), m_rows(0), m_cols(0), m_matrix(nullptr)
    {
        other.swap(*this);
    }

    Matrix& operator=(Matrix other)
    {
        other.swap(*this);
        return *this;
    }

    void swap(Matrix& rhs) noexcept
    {
        using std::swap;
        swap(m_rows, rhs.m_rows);
        swap(m_cols, rhs.m_cols);
        swap(m_dtype, rhs.m_dtype);
        swap(m_matrix, rhs.m_matrix);
    }

    ~Matrix()
    {
        free(m_matrix);
    }

    int get_dtype_size()
    {
        switch (m_dtype) {
        case MatrixType::FLOAT32: return 4;
        case MatrixType::FLOAT64: return 8;
        case MatrixType::INT8: return 1;
        case MatrixType::INT16: return 2;
        case MatrixType::INT32: return 4;
        case MatrixType::INT64: return 8;
        case MatrixType::UINT8: return 1;
        case MatrixType::UINT16: return 2;
        case MatrixType::UINT32: return 4;
        case MatrixType::UINT64: return 8;
        default: throw std::invalid_argument("invalid dtype");
        }
    }

    const char* get_format()
    {
        switch (m_dtype) {
        case MatrixType::FLOAT32: return "f";
        case MatrixType::FLOAT64: return "d";
        case MatrixType::INT8: return "b";
        case MatrixType::INT16: return "h";
        case MatrixType::INT32: return "i";
        case MatrixType::INT64: return "q";
        case MatrixType::UINT8: return "B";
        case MatrixType::UINT16: return "H";
        case MatrixType::UINT32: return "I";
        case MatrixType::UINT64: return "Q";
        default: throw std::invalid_argument("invalid dtype");
        }
    }

    template <typename T>
    void set(size_t row, size_t col, T score)
    {
        void* data = (char*)m_matrix + get_dtype_size() * (row * m_cols + col);
        switch (m_dtype) {
        case MatrixType::FLOAT32: *((float*)data) = (float)score; break;
        case MatrixType::FLOAT64: *((double*)data) = (double)score; break;
        case MatrixType::INT8: *((int8_t*)data) = (int8_t)any_round(score); break;
        case MatrixType::INT16: *((int16_t*)data) = (int16_t)any_round(score); break;
        case MatrixType::INT32: *((int32_t*)data) = (int32_t)any_round(score); break;
        case MatrixType::INT64: *((int64_t*)data) = any_round(score); break;
        case MatrixType::UINT8: *((uint8_t*)data) = (uint8_t)any_round(score); break;
        case MatrixType::UINT16: *((uint16_t*)data) = (uint16_t)any_round(score); break;
        case MatrixType::UINT32: *((uint32_t*)data) = (uint32_t)any_round(score); break;
        case MatrixType::UINT64: *((uint64_t*)data) = (uint64_t)any_round(score); break;
        default: assert(false); break;
        }
    }
};

bool KeyboardInterruptOccured(PyThreadState*& save)
{
    PyEval_RestoreThread(save);
    bool res = PyErr_CheckSignals() != 0;
    save = PyEval_SaveThread();
    return res;
}

template <typename Func>
void run_parallel(int workers, int64_t rows, Func&& func)
{
    PyThreadState* save = PyEval_SaveThread();

    /* for these cases spawning threads causes to much overhead to be worth it */
    if (workers == 0 || workers == 1) {
        for (int64_t row = 0; row < rows; ++row) {
            if (KeyboardInterruptOccured(save)) {
                PyEval_RestoreThread(save);
                throw std::runtime_error("");
            }

            func(row, row + 1);
        }

        PyEval_RestoreThread(save);
        return;
    }

    if (workers < 0) {
        workers = std::thread::hardware_concurrency();
    }

    std::exception_ptr exception = nullptr;
    std::atomic<int> exceptions_occured{0};
    tf::Executor executor(workers);
    tf::Taskflow taskflow;
    std::int64_t step_size = 1;

    taskflow.for_each_index((std::int64_t)0, rows, step_size, [&](std::int64_t row) {
        /* skip work after an exception occured */
        if (exceptions_occured.load() > 0) {
            return;
        }
        try {
            std::int64_t row_end = std::min(row + step_size, rows);
            func(row, row_end);
        }
        catch (...) {
            /* only store first exception */
            if (exceptions_occured.fetch_add(1) == 0) {
                exception = std::current_exception();
            }
        }
    });

    auto future = executor.run(taskflow);
    while (future.wait_for(1s) != std::future_status::ready) {
        if (KeyboardInterruptOccured(save)) {
            exceptions_occured.fetch_add(1);
            future.wait();
            PyEval_RestoreThread(save);
            /* exception already set */
            throw std::runtime_error("");
        }
    }
    PyEval_RestoreThread(save);

    if (exception) std::rethrow_exception(exception);
}

template <typename T>
static Matrix cdist_single_list_impl(const RF_Kwargs* kwargs, RF_Scorer* scorer,
                                     const std::vector<RF_StringWrapper>& queries, MatrixType dtype,
                                     int workers, T score_cutoff)
{
    int64_t rows = queries.size();
    int64_t cols = queries.size();
    Matrix matrix(dtype, static_cast<size_t>(rows), static_cast<size_t>(cols));

    run_parallel(workers, rows, [&](std::int64_t row, std::int64_t row_end) {
        for (; row < row_end; ++row) {
            RF_ScorerFunc scorer_func;
            PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
            RF_ScorerWrapper ScorerFunc(scorer_func);

            T score;
            ScorerFunc.call(&queries[row].string, score_cutoff, &score);
            matrix.set(row, row, score);

            for (int64_t col = row + 1; col < cols; ++col) {
                ScorerFunc.call(&queries[col].string, score_cutoff, &score);
                matrix.set(row, col, score);
                matrix.set(col, row, score);
            }
        }
    });

    return matrix;
}

template <typename T>
static Matrix cdist_two_lists_impl(const RF_Kwargs* kwargs, RF_Scorer* scorer,
                                   const std::vector<RF_StringWrapper>& queries,
                                   const std::vector<RF_StringWrapper>& choices, MatrixType dtype,
                                   int workers, T score_cutoff)
{
    int64_t rows = queries.size();
    int64_t cols = choices.size();
    Matrix matrix(dtype, static_cast<size_t>(rows), static_cast<size_t>(cols));

    run_parallel(workers, rows, [&](std::int64_t row, std::int64_t row_end) {
        for (; row < row_end; ++row) {
            RF_ScorerFunc scorer_func;
            PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
            RF_ScorerWrapper ScorerFunc(scorer_func);

            for (int64_t col = 0; col < cols; ++col) {
                T score;
                ScorerFunc.call(&choices[col].string, score_cutoff, &score);
                matrix.set(row, col, score);
            }
        }
    });

    return matrix;
}
