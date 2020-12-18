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
  // allow different scorers to require different keyword
  /*virtual bool parse_args(PyObject* keywds) {
    // if keywds not empty
  }*/

protected:
  python_string m_str1;
  python_string m_str2;
};


struct PythonStringWrapper {
  PythonStringWrapper(python_string value, PyObject* object = NULL, bool owned = false)
    : value(std::move(value)), object(object), owned(owned) {}

  PythonStringWrapper(const PythonStringWrapper& other) = delete;

  PythonStringWrapper(PythonStringWrapper&& other) {
    value = std::move(other.value);
    object = other.object;
    owned = other.owned;
    other.owned = false;
  }

  PythonStringWrapper& operator=(const PythonStringWrapper& other) = delete;
  PythonStringWrapper& operator=(PythonStringWrapper&& other) {
    value = std::move(other.value);
    object = other.object;
    owned = other.owned;
    other.owned = false;
    return *this;
  }

  ~PythonStringWrapper() {
    if (owned) Py_XDECREF(object);
  }

  python_string value;
  PyObject* object;
  bool owned;
};



struct Processor {
  virtual ~Processor() = default;
  virtual PythonStringWrapper call(PyObject* str, const char* name) = 0;
};

struct NoProcessor : public Processor  {
  NoProcessor() {}
  PythonStringWrapper call(PyObject* str, const char* name) override {
    if (!valid_str(str, name)) throw std::invalid_argument("");
    return PythonStringWrapper(decode_python_string(str), str);
  }
};

struct PythonProcessor : public Processor {
  PythonProcessor(PyObject* processor)
    : m_processor(processor) {}

  PythonStringWrapper call(PyObject* str, const char* name) override {
#if PY_VERSION_HEX >= PYTHON_VERSION(3, 9, 0)
    PyObject* proc_str = PyObject_CallOneArg(m_processor, str);
#else
    PyObject* proc_str = PyObject_CallFunctionObjArgs(m_processor, str, NULL);
#endif
    if ((proc_str == NULL) || (!valid_str(proc_str, name))) {
       throw std::invalid_argument("");
    }
    return PythonStringWrapper(decode_python_string(proc_str), proc_str, true);
  }
private:
  PyObject* m_processor;
};
