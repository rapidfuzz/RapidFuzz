/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "string_metric.hpp"
#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_string_metric.hpp"
#include "py_utils.hpp"

namespace string_metric = rapidfuzz::string_metric;
namespace fuzz = rapidfuzz::fuzz;

// C++11 does not support generic lambdas
template <typename MatchingFunc>
struct GenericRatioVisitor {
  GenericRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return MatchingFunc::call(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

template <typename MatchingFunc>
static inline PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* py_processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|Od", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &py_processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  auto processor = get_processor(py_processor, processor_default);

  try {
    auto s1 = preprocess(py_s1, py_processor, processor, "s1");
    auto s2 = preprocess(py_s2, py_processor, processor, "s2");
    double result = mpark::visit(GenericRatioVisitor<MatchingFunc>(score_cutoff), s1.value, s2.value);
    return PyFloat_FromDouble(result);
  } catch(std::invalid_argument& e) {
    const char* msg = e.what();
    if (msg[0]) {
      PyErr_SetString(PyExc_ValueError, msg);
    }
    return NULL;
  }
}



/**********************************************
 *              Levenshtein
 *********************************************/

struct LevenshteinVisitor {
  LevenshteinVisitor(std::size_t insert_cost, std::size_t delete_cost,
                          std::size_t replace_cost, std::size_t max)
      : m_insert_cost(insert_cost), m_delete_cost(delete_cost), m_replace_cost(replace_cost), m_max(max)
  {}

  template <typename Sentence1, typename Sentence2>
  std::size_t operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return string_metric::levenshtein(s1, s2, {m_insert_cost, m_delete_cost, m_replace_cost}, m_max);
  }

private:
  std::size_t m_insert_cost;
  std::size_t m_delete_cost;
  std::size_t m_replace_cost;
  std::size_t m_max;
};

PyObject* levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* py_weights = NULL;
  PyObject* py_max = NULL;
  Py_ssize_t  max = -1;
  std::size_t insert_cost = 1;
  std::size_t delete_cost = 1;
  std::size_t replace_cost = 1;
  static const char* kwlist[] = {"s1", "s2", "weights", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OO", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &py_weights, &py_max))
  {
    return NULL;
  }

  if (py_max) {
    if (py_max == Py_None) {
      max = -1;
    } else if (!PyLong_Check(py_max)) {
      PyErr_SetString(PyExc_TypeError, "Max must be a integer or None");
      return NULL;
    } else {
      max = PyLong_AsSsize_t(py_max);
      if (PyErr_Occurred()) {
        return NULL;
      }
      if (max < 0) {
        PyErr_SetString(PyExc_TypeError, "Max has to be at least 0");
        return NULL;
      }
    }
  }


  if (py_weights) {
    if (!PyTuple_Check(py_weights)) {
      PyErr_SetString(PyExc_TypeError, "Weights must be a Tuple");
      return NULL;
    }

    if (!PyArg_ParseTuple(py_weights, "nnn", &insert_cost, &delete_cost, &replace_cost)) {
      return NULL;
    }
  }

  // todo improve error message for None
  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);

  std::size_t result = mpark::visit(LevenshteinVisitor(insert_cost, delete_cost, replace_cost, (std::size_t)max),
                                    s1_view, s2_view);

  if (result == (std::size_t)-1) {
    return PyLong_FromLong(-1);
  }
  return PyLong_FromSize_t(result);
}

struct NormalizedLevenshteinVisitor {
  NormalizedLevenshteinVisitor(std::size_t insert_cost, std::size_t delete_cost,
                          std::size_t replace_cost, double score_cutoff)
      : m_insert_cost(insert_cost), m_delete_cost(delete_cost), m_replace_cost(replace_cost),
        m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return string_metric::normalized_levenshtein(
      s1, s2, {m_insert_cost, m_delete_cost, m_replace_cost}, m_score_cutoff);

    // todo catch potential exception somewhere
  }

private:
  std::size_t m_insert_cost;
  std::size_t m_delete_cost;
  std::size_t m_replace_cost;
  double m_score_cutoff;
};


PyObject* normalized_levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* py_weights = NULL;
  std::size_t insert_cost = 1;
  std::size_t delete_cost = 1;
  std::size_t replace_cost = 1;
  PyObject* py_processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "weights", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOd", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &py_weights, &py_processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_weights) {
    if (!PyTuple_Check(py_weights)) {
      PyErr_SetString(PyExc_TypeError, "Weights must be a Tuple");
      return NULL;
    }

    // todo use better error message
    if (!PyArg_ParseTuple(py_weights, "nnn", &insert_cost, &delete_cost, &replace_cost)) {
      return NULL;
    }
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  if (insert_cost != 1 || delete_cost != 1 || replace_cost > 2) {
    PyErr_SetString(PyExc_ValueError, "normalisation for these weightes not supported yet");
    return NULL;
  }

  auto processor = get_processor(py_processor, false);

  try {
    auto s1 = preprocess(py_s1, py_processor, processor, "s1");
    auto s2 = preprocess(py_s2, py_processor, processor, "s2");
    double result = mpark::visit(
      NormalizedLevenshteinVisitor(insert_cost, delete_cost, replace_cost, score_cutoff),
      s1.value, s2.value);

    return PyFloat_FromDouble(result * 100);
  } catch(std::invalid_argument& e) {
    return NULL;
  }
}

/**********************************************
 *                 Hamming
 *********************************************/

struct HammingDistanceVisitor {
  template <typename Sentence1, typename Sentence2>
  std::size_t operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return string_metric::hamming(s1, s2);
  }
};

PyObject* hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  static const char* kwlist[] = {"s1", "s2", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", const_cast<char**>(kwlist), &py_s1, &py_s2))
  {
    return NULL;
  }

  // todo improve error message for None
  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);
  std::size_t result;
  try {
    result = mpark::visit(HammingDistanceVisitor(), s1_view, s2_view);
  } catch(std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return NULL;
  }

  return PyLong_FromSize_t(result);
}

struct normalized_hamming_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return string_metric::normalized_hamming(std::forward<Args>(args)...);
  }
};

PyObject* normalized_hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
  return fuzz_call<normalized_hamming_func>(false, args, keywds);
}

// C++11 does not support generic lambdas
struct NormalizedHammingVisitor {
  NormalizedHammingVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return string_metric::normalized_hamming(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};
