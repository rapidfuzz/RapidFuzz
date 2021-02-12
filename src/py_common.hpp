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

  virtual double ratio(const python_string& s2, double score_cutoff) const = 0;
};

/*
 * this is a generic cached scorer implementation
 * that can be used by all scorers, that do not accept any
 * additional arguments
 */
template <typename Scorer>
struct GenericScorerVisitor {
  GenericScorerVisitor(const Scorer* cached_ratio, double score_cutoff)
    : m_cached_ratio(cached_ratio), m_score_cutoff(score_cutoff) {}

  template <typename Sentence2>
  double operator()(Sentence2&& s2) const {
    return m_cached_ratio->ratio(s2, m_score_cutoff);
  }

private:
  const Scorer* m_cached_ratio;
  double m_score_cutoff;
};

template <template<typename> class Scorer, typename Sentence1>
struct GenericCachedScorer : public CachedScorer {
  GenericCachedScorer(const Sentence1& s1)
    : cached_ratio(s1) {}

  double ratio(const python_string& s2, double score_cutoff) const override {
    return mpark::visit(GenericScorerVisitor<decltype(cached_ratio)>(&cached_ratio, score_cutoff), s2);
  }
private:
  Scorer<Sentence1> cached_ratio;
};


struct PythonStringWrapper {
  PythonStringWrapper(python_string value, PyObject* object = NULL, bool owned = false)
    : value(std::move(value)), object(object), owned(owned) {}

  PythonStringWrapper(const PythonStringWrapper& other) = delete;

  PythonStringWrapper(PythonStringWrapper&& other) {
    std::swap(value, other.value);
    object = other.object;
    owned = other.owned;
    other.owned = false;
  }

  PythonStringWrapper& operator=(const PythonStringWrapper& other) = delete;
  PythonStringWrapper& operator=(PythonStringWrapper&& other) {
    std::swap(value, other.value);
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

struct PythonProcessor {
  static PythonStringWrapper call(PyObject* processor, PyObject* str, const char* name) {
#if PY_VERSION_HEX >= PYTHON_VERSION(3, 9, 0)
    PyObject* proc_str = PyObject_CallOneArg(processor, str);
#else
    PyObject* proc_str = PyObject_CallFunctionObjArgs(processor, str, NULL);
#endif
    if ((proc_str == NULL) || (!valid_str(proc_str, name))) {
       throw std::invalid_argument("");
    }
    return PythonStringWrapper(decode_python_string(proc_str), proc_str, true);
  }
};
