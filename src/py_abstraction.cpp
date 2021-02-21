/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "py_string_metric.hpp"
#include "py_utils.hpp"


static PyMethodDef methods[] = {
    /* utils */
    PY_METHOD(default_process),
    /* string_metric */
    PY_METHOD(levenshtein),
    PY_METHOD(normalized_levenshtein),
    PY_METHOD(hamming),
    PY_METHOD(normalized_hamming),
    /* fuzz */
    PY_METHOD(ratio),
    PY_METHOD(partial_ratio),
    PY_METHOD(token_sort_ratio),
    PY_METHOD(partial_token_sort_ratio),
    PY_METHOD(token_set_ratio),
    PY_METHOD(partial_token_set_ratio),
    PY_METHOD(token_ratio),
    PY_METHOD(partial_token_ratio),
    PY_METHOD(WRatio),
    PY_METHOD(QRatio),
    /* sentinel */
    {NULL, NULL, 0, NULL}};


#if PY_VERSION_HEX < PYTHON_VERSION(3, 0, 0)

PyMODINIT_FUNC initcpp_impl(void)
{
  Py_InitModule3("cpp_impl", methods, NULL);
}

#else

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpp_impl",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_cpp_impl(void)
{
  return PyModule_Create(&moduledef);
}

#endif
