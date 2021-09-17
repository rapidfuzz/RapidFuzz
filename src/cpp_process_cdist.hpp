#pragma once
#include "cpp_process.hpp"
#include "numpy/ndarraytypes.h"

static PyObject* cdist_single_list_distance_impl(
    const KwargsContext& kwargs_context, distance_context_init init,
    const std::vector<proc_string>& queries, size_t max)
{
    std::size_t rows = queries.size();
    std::size_t cols = queries.size();
    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);

    int32_t* data = (int32_t*)matrix->data;

Py_BEGIN_ALLOW_THREADS
    for (size_t row = 0; row < rows; ++row)
    {
        data[row*cols + row] = 0;
        CachedDistanceContext DistanceContext = init(kwargs_context, queries[row]);
        for (size_t col = row + 1; col < cols; ++col)
        {
            int32_t score = (int32_t)DistanceContext.ratio(queries[col], max);
            data[row*cols + col] = score;
            data[col*cols + row] = score;
        }
    }
Py_END_ALLOW_THREADS

    return (PyObject*)matrix;
}

static PyObject* cdist_single_list_similarity_impl(
    const KwargsContext& kwargs_context, scorer_context_init init,
    const std::vector<proc_string>& queries, double score_cutoff)
{
    std::size_t rows = queries.size();
    std::size_t cols = queries.size();
    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT8);

    uint8_t* data = (uint8_t*)matrix->data;
    score_cutoff = std::floor(score_cutoff);
    
Py_BEGIN_ALLOW_THREADS
    for (size_t row = 0; row < rows; ++row)
    {
        data[row*cols + row] = 100;
        CachedScorerContext ScorerContext = init(kwargs_context, queries[row]);
        for (size_t col = row + 1; col < cols; ++col)
        {
            uint8_t score = (uint8_t)std::floor(ScorerContext.ratio(queries[col], score_cutoff));
            data[row*cols + col] = score;
            data[col*cols + row] = score;
        }
    }
Py_END_ALLOW_THREADS

    return (PyObject*)matrix;
}


static PyObject* cdist_two_lists_distance_impl(
    const KwargsContext& kwargs_context, distance_context_init init,
    const std::vector<proc_string>& queries, const std::vector<proc_string>& choices, size_t max)
{
    std::size_t rows = queries.size();
    std::size_t cols = choices.size();
    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT32);

    int32_t* data = (int32_t*)matrix->data;

Py_BEGIN_ALLOW_THREADS
    for (size_t row = 0; row < rows; ++row)
    {
        CachedDistanceContext DistanceContext = init(kwargs_context, queries[row]);
        for (size_t col = 0; col < cols; ++col)
        {
            data[row*cols + col] = (int32_t)DistanceContext.ratio(choices[col], max);
        }
    }
Py_END_ALLOW_THREADS

    return (PyObject*)matrix;
}

static PyObject* cdist_two_lists_similarity_impl(
    const KwargsContext& kwargs_context, scorer_context_init init,
    const std::vector<proc_string>& queries, const std::vector<proc_string>& choices, double score_cutoff)
{
    std::size_t rows = queries.size();
    std::size_t cols = choices.size();

    npy_intp dims[] = {(npy_intp)rows, (npy_intp)cols};
    PyArrayObject* matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT8);

    uint8_t* data = (uint8_t*)matrix->data;
    score_cutoff = std::floor(score_cutoff);

Py_BEGIN_ALLOW_THREADS
    for (size_t row = 0; row < rows; ++row)
    {
        CachedScorerContext ScorerContext = init(kwargs_context, queries[row]);
        for (size_t col = 0; col < cols; ++col)
        {
            data[row*cols + col] = (uint8_t)std::floor(ScorerContext.ratio(choices[col], score_cutoff));
        }
    }
Py_END_ALLOW_THREADS

    return (PyObject*)matrix;
}
