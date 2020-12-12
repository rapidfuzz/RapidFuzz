/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "levenshtein.hpp"
#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_string_metric.hpp"
#include "py_utils.hpp"

namespace rlevenshtein = rapidfuzz::levenshtein;
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
    auto s1 = processor->call(py_s1, "s1");
    auto s2 = processor->call(py_s2, "s2");
    double result = mpark::visit(GenericRatioVisitor<MatchingFunc>(score_cutoff), s1.value, s2.value);
    return PyFloat_FromDouble(result);
  } catch(std::invalid_argument& e) {
    return NULL;
  }
}



/**********************************************
 *              Levenshtein
 *********************************************/

struct LevenshteinVisitor {
  LevenshteinVisitor(std::size_t insert_cost, std::size_t delete_cost,
                          std::size_t replace_cost)
      : m_insert_cost(insert_cost), m_delete_cost(delete_cost), m_replace_cost(replace_cost)
  {}

  template <typename Sentence1, typename Sentence2>
  std::size_t operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    if (m_insert_cost == 1 && m_delete_cost == 1) {
      if (m_replace_cost == 1) {
        return rlevenshtein::distance(s1, s2);
      }
      else if (m_replace_cost == 2) {
        return rlevenshtein::weighted_distance(s1, s2);
      }
    }

    return rlevenshtein::generic_distance(s1, s2, {m_insert_cost, m_delete_cost, m_replace_cost});
  }

private:
  std::size_t m_insert_cost;
  std::size_t m_delete_cost;
  std::size_t m_replace_cost;
};

PyObject* levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  std::size_t insert_cost = 1;
  std::size_t delete_cost = 1;
  std::size_t replace_cost = 1;
  static const char* kwlist[] = {"s1", "s2", "insert_cost", "delete_cost", "replace_cost", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|nnn", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &insert_cost, &delete_cost, &replace_cost))
  {
    return NULL;
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }
  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);

  std::size_t result = mpark::visit(LevenshteinVisitor(insert_cost, delete_cost, replace_cost),
                                    s1_view, s2_view);
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
    if (m_insert_cost == 1 && m_delete_cost == 1) {
      if (m_replace_cost == 1) {
        return rlevenshtein::normalized_distance(s1, s2, m_score_cutoff);
      }
      else if (m_replace_cost == 2) {
        return rlevenshtein::normalized_weighted_distance(s1, s2, m_score_cutoff);
      }
    }
    /* todo there is no normalized version of this yet */
    return 0.0;
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
  std::size_t insert_cost = 1;
  std::size_t delete_cost = 1;
  std::size_t replace_cost = 1;
  PyObject* py_processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "insert_cost", "delete_cost", "replace_cost", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|nnnOd", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &insert_cost, &delete_cost, &replace_cost, &py_processor, &score_cutoff))
  {
    return NULL;
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
    auto s1 = processor->call(py_s1, "s1");
    auto s2 = processor->call(py_s2, "s2");
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
    return rlevenshtein::hamming(s1, s2);
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


/**********************************************
 *         normalized_letter_frequency
 *********************************************/

struct normalized_letter_frequency_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::quick_lev_ratio(std::forward<Args>(args)...);
  }
};

PyObject* normalized_letter_frequency(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
  return fuzz_call<normalized_letter_frequency_func>(false, args, keywds);
}

// C++11 does not support generic lambdas
struct NormalizedLetterFrequencyVisitor {
  NormalizedLetterFrequencyVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return fuzz::quick_lev_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedNormalizedLetterFrequency::call(double score_cutoff) {
  return mpark::visit(NormalizedLetterFrequencyVisitor(score_cutoff), m_str1, m_str2);
}