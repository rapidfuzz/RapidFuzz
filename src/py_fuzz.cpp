/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "fuzz.hpp"
#include "string_utils.hpp"
#include <boost/utility/string_view.hpp>

namespace fuzz = rapidfuzz::fuzz;

static PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::partial_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::token_sort_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::partial_token_sort_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::token_set_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::partial_token_set_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::token_ratio(
            rapidfuzz::Sentence<wchar_t>(s1),
            rapidfuzz::Sentence<wchar_t>(s2),
            score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|d", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::partial_token_ratio(s1, s2, score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    long long s1_bitmap = 0;
    long long s2_bitmap = 0;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "s1_bitmap", "s2_bitmap", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|LLd", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &s1_bitmap, &s2_bitmap, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::WRatio(
            rapidfuzz::Sentence<wchar_t>(s1, s1_bitmap),
            rapidfuzz::Sentence<wchar_t>(s2, s2_bitmap),
            score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}

static PyObject* bitmap_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    long long s1_bitmap = 0;
    long long s2_bitmap = 0;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "s1_bitmap", "s2_bitmap", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|LLd", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &s1_bitmap, &s2_bitmap, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    Py_ssize_t len_s1 = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer_s1 = PyUnicode_AsWideCharString(py_s1, &len_s1);
    boost::wstring_view s1(buffer_s1, len_s1);
    
    Py_ssize_t len_s2 = PyUnicode_GET_LENGTH(py_s2);
    wchar_t* buffer_s2 = PyUnicode_AsWideCharString(py_s2, &len_s2);
    boost::wstring_view s2(buffer_s2, len_s2);

    double result = fuzz::bitmap_ratio(
            rapidfuzz::Sentence<wchar_t>(s1, s1_bitmap),
            rapidfuzz::Sentence<wchar_t>(s2, s2_bitmap),
            score_cutoff);
    
    PyMem_Free(buffer_s1);
    PyMem_Free(buffer_s2);

    return PyFloat_FromDouble(result);
}


/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/
#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, "" }
static PyMethodDef methods[] = {
    PY_METHOD(ratio),
    PY_METHOD(partial_ratio),
    PY_METHOD(token_sort_ratio),
    PY_METHOD(partial_token_sort_ratio),
    PY_METHOD(token_set_ratio),
    PY_METHOD(partial_token_set_ratio),
    PY_METHOD(token_ratio),
    PY_METHOD(partial_token_ratio),
    PY_METHOD(WRatio),
    PY_METHOD(bitmap_ratio),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz._fuzz",
    NULL,
    -1,
    methods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    0,     /* m_clear */
    NULL   /* m_free */
};

PyMODINIT_FUNC PyInit__fuzz(void) {
    return PyModule_Create(&moduledef);
}