/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "utils.hpp"

namespace utils = rapidfuzz::utils;

constexpr const char * default_process_docstring = R"()";

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
    void* str = PyUnicode_DATA(py_sentence);

    int str_kind = PyUnicode_KIND(py_sentence);

    PyObject* result;

    switch (str_kind) {
    case PyUnicode_1BYTE_KIND:
    {
        auto proc_str = utils::default_process(nonstd::basic_string_view<uint8_t>(static_cast<uint8_t*>(str), len));
        result = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, proc_str.data(), proc_str.size());
        break;
    }
    case PyUnicode_2BYTE_KIND:
    {
        auto proc_str = utils::default_process(nonstd::basic_string_view<uint16_t>(static_cast<uint16_t*>(str), len));
        result = PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, proc_str.data(), proc_str.size());
        break;
    }
    default:
    {
        auto proc_str = utils::default_process(nonstd::basic_string_view<uint32_t>(static_cast<uint32_t*>(str), len));
        result = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, proc_str.data(), proc_str.size());
        break;
    }
    }

    return result;

}

/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/
#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, x##_docstring }
static PyMethodDef methods[] = {
    PY_METHOD(default_process),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz.utils",
    NULL,
    -1,
    methods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    0,     /* m_clear */
    NULL   /* m_free */
};

PyMODINIT_FUNC PyInit_utils(void) {
    return PyModule_Create(&moduledef);
}
