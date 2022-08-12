#pragma once
#include "process_cpp.hpp"
#include "taskflow/taskflow.hpp"
#include "rapidfuzz_capi.h"
#include <exception>
#include <atomic>
#include <chrono>
using namespace std::chrono_literals;

int64_t any_round(double score)
{
    return std::llround(score);
}

int64_t any_round(int64_t score)
{
    return score;
}

enum class MatrixType
{
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

struct Matrix
{
    MatrixType m_dtype;
    size_t m_rows;
    size_t m_cols;
    void* m_matrix;

    Matrix() : m_dtype(MatrixType::FLOAT32), m_rows(0), m_cols(0), m_matrix(nullptr) {}

    Matrix(MatrixType dtype, size_t rows, size_t cols)
        : m_dtype(dtype), m_rows(rows), m_cols(cols)
    {

        m_matrix = malloc(get_dtype_size() * m_rows * m_cols);
        if (m_matrix == nullptr)
            throw std::bad_alloc();
    }

    Matrix(const Matrix& other) : m_dtype(other.m_dtype), m_rows(other.m_rows), m_cols(other.m_cols)
    {
        m_matrix = malloc(get_dtype_size() * m_rows * m_cols);
        if (m_matrix == nullptr)
            throw std::bad_alloc();

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
        switch(m_dtype)
        {
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
        default:
            throw std::invalid_argument("invalid dtype");
        }
    }

    const char* get_format()
    {
        switch(m_dtype)
        {
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
        default:
            throw std::invalid_argument("invalid dtype");
        }
    }

    template <typename T>
    void set(size_t row, size_t col, T score)
    {
        void* data = (char*)m_matrix + get_dtype_size() * (row * m_cols + col);
        switch(m_dtype)
        {
        case MatrixType::FLOAT32:
            *((float*)data) = (float)score;
            break;
        case MatrixType::FLOAT64:
            *((double*)data) = (double)score;
            break;
        case MatrixType::INT8:
            *((int8_t*)data) = (int8_t)any_round(score);
            break;
        case MatrixType::INT16:
            *((int16_t*)data) = (int16_t)any_round(score);
            break;
        case MatrixType::INT32:
            *((int32_t*)data) = (int32_t)any_round(score);
            break;
        case MatrixType::INT64:
            *((int64_t*)data) = any_round(score);
            break;
        case MatrixType::UINT8:
            *((uint8_t*)data) = (uint8_t)any_round(score);
            break;
        case MatrixType::UINT16:
            *((uint16_t*)data) = (uint16_t)any_round(score);
            break;
        case MatrixType::UINT32:
            *((uint32_t*)data) = (uint32_t)any_round(score);
            break;
        case MatrixType::UINT64:
            *((uint64_t*)data) = (uint64_t)any_round(score);
            break;
        default:
            assert(false);
            break;
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
    if (workers == 0 || workers == 1)
    {
        for (int64_t row = 0; row < rows; ++row)
        {
            if (KeyboardInterruptOccured(save))
            {
                PyEval_RestoreThread(save);
                throw std::runtime_error("");
            }

            func(row, row + 1);
        }

        PyEval_RestoreThread(save);
        return;
    }

    if (workers < 0)
    {
        workers = std::thread::hardware_concurrency();
    }

    std::exception_ptr exception = nullptr;
    std::atomic<int> exceptions_occured{0};
    tf::Executor executor(workers);
    tf::Taskflow taskflow;
    std::int64_t step_size = 1;

    taskflow.for_each_index((std::int64_t)0, rows, step_size, [&] (std::int64_t row) {
        /* skip work after an exception occured */
        if (exceptions_occured.load() > 0) {
            return;
        }
        try
        {
            std::int64_t row_end = std::min(row + step_size, rows);
            func(row, row_end);
        }
        catch(...)
        {
            /* only store first exception */
            if (exceptions_occured.fetch_add(1) == 0) {
                exception = std::current_exception();
            }
        }
    });

    auto future = executor.run(taskflow);
    while (future.wait_for(1s) != std::future_status::ready)
    {
        if (KeyboardInterruptOccured(save))
        {
            exceptions_occured.fetch_add(1);
            future.wait();
            PyEval_RestoreThread(save);
            /* exception already set */
            throw std::runtime_error("");
        }
    }
    PyEval_RestoreThread(save);

    if (exception)
        std::rethrow_exception(exception);
}

template <typename T>
static Matrix cdist_single_list_impl(
    const RF_Kwargs* kwargs, RF_Scorer* scorer,
    const std::vector<RF_StringWrapper>& queries, MatrixType dtype, int workers, T score_cutoff)
{
    int64_t rows = queries.size();
    int64_t cols = queries.size();
    Matrix matrix(dtype, static_cast<size_t>(rows), static_cast<size_t>(cols));

    run_parallel(workers, rows, [&] (std::int64_t row, std::int64_t row_end) {
        for (; row < row_end; ++row)
        {
            RF_ScorerFunc scorer_func;
            PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
            RF_ScorerWrapper ScorerFunc(scorer_func);

            T score;
            ScorerFunc.call(&queries[row].string, score_cutoff, &score);
            matrix.set(row, row, score);

            for (int64_t col = row + 1; col < cols; ++col)
            {
                ScorerFunc.call(&queries[col].string, score_cutoff, &score);
                matrix.set(row, col, score);
                matrix.set(col, row, score);
            }
        }
    });

    return matrix;
}

template <typename T>
static Matrix cdist_two_lists_impl(
    const RF_Kwargs* kwargs, RF_Scorer* scorer,
    const std::vector<RF_StringWrapper>& queries, const std::vector<RF_StringWrapper>& choices, MatrixType dtype, int workers, T score_cutoff)
{
    int64_t rows = queries.size();
    int64_t cols = choices.size();
    Matrix matrix(dtype, static_cast<size_t>(rows), static_cast<size_t>(cols));

    run_parallel(workers, rows, [&] (std::int64_t row, std::int64_t row_end) {
        for (; row < row_end; ++row)
        {
            RF_ScorerFunc scorer_func;
            PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
            RF_ScorerWrapper ScorerFunc(scorer_func);

            for (int64_t col = 0; col < cols; ++col)
            {
                T score;
                ScorerFunc.call(&choices[col].string, score_cutoff, &score);
                matrix.set(row, col, score);
            }
        }
    });

    return matrix;
}
