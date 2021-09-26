#pragma once
#include "cpp_process.hpp"
#include "numpy/ndarraytypes.h"
#include "taskflow/taskflow.hpp"
#include <exception>
#include <atomic>

void set_score_distance(PyArrayObject* matrix, int dtype, npy_intp row, npy_intp col, size_t score)
{
    void* data = PyArray_GETPTR2(matrix, row, col);
    switch (dtype)
    {
    case NPY_INT8:
        *((int8_t*)data) = (int8_t)score;
        break;
    case NPY_INT16:
        *((int16_t*)data) = (int16_t)score;
        break;
    case NPY_INT32:
        *((int32_t*)data) = (int32_t)score;
        break;
    case NPY_INT64:
        *((int64_t*)data) = (int64_t)score;
        break;
    default:
        assert(false);
        break;
    }
}

void set_score_similarity(PyArrayObject* matrix, int dtype, npy_intp row, npy_intp col, double score)
{
    void* data = PyArray_GETPTR2(matrix, row, col);
    switch (dtype)
    {
    case NPY_UINT8:
        *((uint8_t*)data) = (uint8_t)std::round(score);
        break;
    case NPY_FLOAT32:
        *((float*)data) = (float)score;
        break;
    case NPY_FLOAT64:
        *((double*)data) = score;
        break;
    default:
        assert(false);
        break;
    }
}

template <typename Func>
void run_parallel(int workers, size_t rows, Func&& func)
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
    std::size_t step_size = 1;

    taskflow.for_each_index((std::size_t)0, rows, step_size, [&] (std::size_t row) {
        /* skip work after an exception occured */
        if (exceptions_occured.load() > 0) {
            return;
        }
        try
        {
            std::size_t row_end = std::min(row + step_size, rows);
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

static PyObject* cdist_single_list_distance_impl(
    const KwargsContext& kwargs_context, distance_context_init init,
    const std::vector<proc_string>& queries, int dtype, int workers, size_t max)
{
    std::size_t rows = queries.size();
    std::size_t cols = queries.size();
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
        run_parallel(workers, rows, [&] (std::size_t row, std::size_t row_end) {
            for (; row < row_end; ++row)
            {
                set_score_distance(matrix, dtype, row, row, 0);
                CachedDistanceContext DistanceContext = init(kwargs_context, queries[row]);
                for (size_t col = row + 1; col < cols; ++col)
                {
                    size_t score = DistanceContext.ratio(queries[col], max);
                    set_score_distance(matrix, dtype, row, col, score);
                    set_score_distance(matrix, dtype, col, row, score);
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

static PyObject* cdist_single_list_similarity_impl(
    const KwargsContext& kwargs_context, scorer_context_init init,
    const std::vector<proc_string>& queries, int dtype, int workers, double score_cutoff)
{
    std::size_t rows = queries.size();
    std::size_t cols = queries.size();
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
        run_parallel(workers, rows, [&] (std::size_t row, std::size_t row_end) {
            for (; row < row_end; ++row)
            {
                set_score_similarity(matrix, dtype, row, row, 100);
                CachedScorerContext ScorerContext = init(kwargs_context, queries[row]);
                for (size_t col = row + 1; col < cols; ++col)
                {
                    double score = ScorerContext.ratio(queries[col], score_cutoff);
                    set_score_similarity(matrix, dtype, row, col, score);
                    set_score_similarity(matrix, dtype, col, row, score);
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

static PyObject* cdist_two_lists_distance_impl(
    const KwargsContext& kwargs_context, distance_context_init init,
    const std::vector<proc_string>& queries, const std::vector<proc_string>& choices, int dtype, int workers, size_t max)
{
    std::size_t rows = queries.size();
    std::size_t cols = choices.size();
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
        run_parallel(workers, rows, [&] (std::size_t row, std::size_t row_end) {
            for (; row < row_end; ++row)
            {
                CachedDistanceContext DistanceContext = init(kwargs_context, queries[row]);
                for (size_t col = 0; col < cols; ++col)
                {
                    size_t score = (int)DistanceContext.ratio(choices[col], max);
                    set_score_distance(matrix, dtype, row, col, score);
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

static PyObject* cdist_two_lists_similarity_impl(
    const KwargsContext& kwargs_context, scorer_context_init init,
    const std::vector<proc_string>& queries, const std::vector<proc_string>& choices, int dtype, int workers, double score_cutoff)
{
    std::size_t rows = queries.size();
    std::size_t cols = choices.size();

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
        run_parallel(workers, rows, [&] (std::size_t row, std::size_t row_end) {
            for (; row < row_end; ++row)
            {
                CachedScorerContext ScorerContext = init(kwargs_context, queries[row]);
                for (size_t col = 0; col < cols; ++col)
                {
                    double score = ScorerContext.ratio(choices[col], score_cutoff);
                    set_score_similarity(matrix, dtype, row, col, score);
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
