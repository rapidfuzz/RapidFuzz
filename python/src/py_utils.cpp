#pragma once
#include <Python.h>
#include <string>
#include "utils.hpp"

constexpr const char * bitmap_create_docstring = R"(

)";

static PyObject* bitmap_create(PyObject *self, PyObject *args, PyObject *keywds) {
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

    return PyLong_FromLong(result);
}

/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/
#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, x##_docstring }
static PyMethodDef methods[] = {
    PY_METHOD(bitmap_create),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz._utils",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__utils(void) {
    return PyModule_Create(&moduledef);
}
