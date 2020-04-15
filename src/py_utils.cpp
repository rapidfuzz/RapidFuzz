/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "utils.hpp"
#include "string_utils.hpp"

namespace utils = rapidfuzz::utils;
namespace string_utils = rapidfuzz::string_utils;

constexpr const char * bitmap_create_docstring = R"(

)";

static PyObject* bitmap_create(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    static const char *kwlist[] = {"s1", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "U", const_cast<char **>(kwlist),
                                     &py_s1)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1)) {
        return NULL;
    }

    Py_ssize_t len = PyUnicode_GET_LENGTH(py_s1);
    wchar_t* buffer = PyUnicode_AsWideCharString(py_s1, &len);
    uint64_t result = utils::bitmap_create(std::wstring(buffer, len));
    PyMem_Free(buffer);

    return PyLong_FromUnsignedLongLong(result);
}


constexpr const char * default_process_docstring = R"(

)";

static PyObject* default_process(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    PyObject *py_sentence;
    static const char *kwlist[] = {"sentence", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "U", const_cast<char **>(kwlist),
                                     &py_sentence)) {
        return NULL;
    }

    if (PyUnicode_READY(py_sentence)) {
        return NULL;
    }

    Py_ssize_t len = PyUnicode_GET_LENGTH(py_sentence);
    wchar_t* buffer = PyUnicode_AsWideCharString(py_sentence, &len);
    std::wstring result = string_utils::default_process(std::wstring(buffer, len));
    PyMem_Free(buffer);

    return PyUnicode_FromWideChar(result.c_str(), result.length());
}

/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/
#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, x##_docstring }
static PyMethodDef methods[] = {
    PY_METHOD(bitmap_create),
    PY_METHOD(default_process),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz._utils",
    NULL,
    -1,
    methods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    0,     /* m_clear */
    NULL   /* m_free */
};

PyMODINIT_FUNC PyInit__utils(void) {
    return PyModule_Create(&moduledef);
}
