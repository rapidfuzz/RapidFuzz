#pragma once
#include "cpp_process.hpp"
#include "numpy/ndarraytypes.h"
#include "taskflow/taskflow.hpp"
#include "rapidfuzz_capi.h"
#include <exception>
#include <atomic>

int64_t any_round(double score)
{
    return std::llround(score);
}

int64_t any_round(int64_t score)
{
    return score;
}

template <typename T>
void set_score(PyArrayObject* matrix, int dtype, npy_intp row, npy_intp col, T score)
{
    void* data = PyArray_GETPTR2(matrix, row, col);
    switch (dtype)
    {
    case NPY_FLOAT32:
        *((float*)data) = (float)score;
        break;
    case NPY_FLOAT64:
        *((double*)data) = score;
        break;
    case NPY_INT8:
        *((int8_t*)data) = (int8_t)any_round(score);
        break;
    case NPY_INT16:
        *((int16_t*)data) = (int16_t)any_round(score);
        break;
    case NPY_INT32:
        *((int32_t*)data) = (int32_t)any_round(score);
        break;
    case NPY_INT64:
        *((int64_t*)data) = any_round(score);
        break;
    case NPY_UINT8:
        *((uint8_t*)data) = (uint8_t)any_round(score);
        break;
    case NPY_UINT16:
        *((uint16_t*)data) = (uint16_t)any_round(score);
        break;
    case NPY_UINT32:
        *((uint32_t*)data) = (uint32_t)any_round(score);
        break;
    case NPY_UINT64:
        *((uint64_t*)data) = (uint64_t)any_round(score);
        break;
    default:
        assert(false);
        break;
    }
}

template <typename Func>
void run_parallel(int workers, int64_t rows, Func&& func)
{
    /* for these cases spawning threads causes to much overhead to be worth it */
    if (workers == 0 || workers == 1)
    {
        func(0, rows);
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

    executor.run(taskflow).get();

    if (exception) {
        std::rethrow_exception(exception);
    }
}

template <typename T>
static PyObject* cdist_single_list_impl(
    const RF_Kwargs* kwargs, RF_Scorer* scorer,
    const std::vector<RF_StringWrapper>& queries, int dtype, int workers, T score_cutoff)
{
    std::int64_t rows = queries.size();
    std::int64_t cols = queries.size();
    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, dtype);

    if (matrix == NULL)
    {
        return NULL;
    }

    std::exception_ptr exception = nullptr;

Py_BEGIN_ALLOW_THREADS
    try
    {
        run_parallel(workers, rows, [&] (std::int64_t row, std::int64_t row_end) {
            for (; row < row_end; ++row)
            {
                RF_ScorerFunc scorer_func;
                PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
                RF_ScorerWrapper ScorerFunc(scorer_func);

                T score;
                ScorerFunc.call(&queries[row].string, score_cutoff, &score);
                set_score(matrix, dtype, row, row, score);

                for (int64_t col = row + 1; col < cols; ++col)
                {
                    ScorerFunc.call(&queries[col].string, score_cutoff, &score);
                    set_score(matrix, dtype, row, col, score);
                    set_score(matrix, dtype, col, row, score);
                }
            }
        });
    }
    catch(...)
    {
        exception = std::current_exception();
    }
Py_END_ALLOW_THREADS

    if (exception) {
        std::rethrow_exception(exception);
    }

    return (PyObject*)matrix;
}

template <typename T>
static PyObject* cdist_two_lists_impl(
    const RF_Kwargs* kwargs, RF_Scorer* scorer,
    const std::vector<RF_StringWrapper>& queries, const std::vector<RF_StringWrapper>& choices, int dtype, int workers, T score_cutoff)
{
    std::int64_t rows = queries.size();
    std::int64_t cols = choices.size();
    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, dtype);

    if (matrix == NULL)
    {
        return NULL;
    }

    std::exception_ptr exception = nullptr;

Py_BEGIN_ALLOW_THREADS
    try
    {
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
                    set_score(matrix, dtype, row, col, score);
                }
            }
        });
    }
    catch(...)
    {
        exception = std::current_exception();
    }
Py_END_ALLOW_THREADS

    if (exception) {
        std::rethrow_exception(exception);
    }

    return (PyObject*)matrix;
}
