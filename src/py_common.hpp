/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#define PY_SSIZE_T_CLEAN
#include "utils.hpp"
#include <Python.h>
#include <vector>

#define PY_FUNC_CAST(func) ((PyCFunction)(void (*)(void))func)

#define PYTHON_VERSION(major, minor, micro) ((major << 24) | (minor << 16) | (micro << 8))

/* The cast of the function is necessary since PyCFunction values
 * only take two PyObject* parameters, and these functions take three.
 */
#define PY_METHOD(x) { #x, PY_FUNC_CAST(x), METH_VARARGS | METH_KEYWORDS, x##_docstring }

#if PY_VERSION_HEX < PYTHON_VERSION(3, 0, 0)
#  define PYTHON_2
#  include "py2_common.hpp"
#else
#  define PYTHON_3
#  include "py3_common.hpp"
#endif


struct CachedScorer {
  virtual ~CachedScorer() = default;

  virtual void str1_set(python_string str)
  {
    m_str1 = std::move(str);
  }

  virtual void str2_set(python_string str)
  {
    m_str2 = std::move(str);
  }

  virtual double call(double score_cutoff) = 0;

protected:
  python_string m_str1;
  python_string m_str2;
};
