/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "levenshtein.hpp"
#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_string_metric.hpp"

namespace rlevenshtein = rapidfuzz::levenshtein;
namespace fuzz = rapidfuzz::fuzz;

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
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "insert_cost", "delete_cost", "replace_cost", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|nnnd", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &insert_cost, &delete_cost, &replace_cost, &score_cutoff))
  {
    return NULL;
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }
  
  if (insert_cost != 1 || delete_cost != 1 || replace_cost > 2) {
    PyErr_SetString(PyExc_ValueError, "normalisation for these weightes not supported yet");
    return NULL;
  }

  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);

  double result = mpark::visit(
    NormalizedLevenshteinVisitor(insert_cost, delete_cost, replace_cost, score_cutoff),
    s1_view, s2_view);

  return PyFloat_FromDouble(result * 100);
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
 *              quick_lev_ratio
 *********************************************/
// todo
PyObject* quick_lev_ratio(PyObject* /*self*/, PyObject* /*args*/, PyObject* /*keywds*/) {
  return NULL;
}

// C++11 does not support generic lambdas
struct QuickLevRatioVisitor {
  QuickLevRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return fuzz::quick_lev_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedQuickLevRatio::call(double score_cutoff) {
  return mpark::visit(QuickLevRatioVisitor(score_cutoff), m_str1, m_str2);
}