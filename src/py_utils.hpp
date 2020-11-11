/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include "utils.hpp"

#define PY_FUNC_CAST(func) ((PyCFunction)(void (*)(void))func)

/* The cast of the function is necessary since PyCFunction values
 * only take two PyObject* parameters, and these functions take three.
 */
#define PY_METHOD(x) \
    { #x, PY_FUNC_CAST(x), METH_VARARGS | METH_KEYWORDS, x##_docstring }

#if PY_MAJOR_VERSION == 2
#define PYTHON_2
#include "py2_utils.hpp"
#else
#define PYTHON_3
#include "py3_utils.hpp"
#endif
