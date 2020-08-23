/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "levenshtein.hpp"
#include "py_utils.hpp"

namespace rlevenshtein = rapidfuzz::levenshtein;

constexpr const char* distance_docstring = R"(
Calculates the minimum number of insertions, deletions, and substitutions
required to change one sequence into the other according to Levenshtein.

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare

Returns:
    int: levenshtein distance between s1 and s2
)";

PyObject* distance(PyObject* /*self*/, PyObject* args, PyObject* keywds)
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
  std::size_t result =
      mpark::visit([](auto&& val1, auto&& val2) { return rlevenshtein::distance(val1, val2); },
                   s1_view, s2_view);

  return PyLong_FromSize_t(result);
}

constexpr const char* normalized_distance_docstring = R"(
Calculates a normalized levenshtein distance based on levenshtein.distance

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.

Returns:
    float: normalized levenshtein distance between s1 and s2 as a float between 0 and 100
)";

PyObject* normalized_distance(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|d", const_cast<char**>(kwlist), &py_s1, &py_s2,
                                   &score_cutoff))
  {
    return NULL;
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);
  double result = mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
        return rlevenshtein::normalized_distance(val1, val2, score_cutoff / 100);
      },
      s1_view, s2_view);

  return PyFloat_FromDouble(result * 100);
}

constexpr const char* weighted_distance_docstring = R"(
Calculates the minimum number of insertions, deletions, and substitutions
required to change one sequence into the other according to Levenshtein with custom
costs for insertion, deletion and substitution

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    insert_cost (int): cost for insertions
    delete_cost (int): cost for deletions
    replace_cost (int): cost for substitutions

Returns:
    int: weighted levenshtein distance between s1 and s2
)";

PyObject* weighted_distance(PyObject* /*self*/, PyObject* args, PyObject* keywds)
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

  std::size_t result = 0;
  if (insert_cost == 1 && delete_cost == 1) {
    if (replace_cost == 1) {
      result =
          mpark::visit([](auto&& val1, auto&& val2) { return rlevenshtein::distance(val1, val2); },
                       s1_view, s2_view);
    }
    else if (replace_cost == 2) {
      result = mpark::visit(
          [](auto&& val1, auto&& val2) { return rlevenshtein::weighted_distance(val1, val2); },
          s1_view, s2_view);
    }
    else {
      result = mpark::visit(
          [insert_cost, delete_cost, replace_cost](auto&& val1, auto&& val2) {
            return rlevenshtein::generic_distance(val1, val2,
                                                  {insert_cost, delete_cost, replace_cost});
          },
          s1_view, s2_view);
    }
  }
  else {
    result = mpark::visit(
        [insert_cost, delete_cost, replace_cost](auto&& val1, auto&& val2) {
          return rlevenshtein::generic_distance(val1, val2,
                                                {insert_cost, delete_cost, replace_cost});
        },
        s1_view, s2_view);
  }

  return PyLong_FromSize_t(result);
}

constexpr const char* normalized_weighted_distance_docstring = R"(
Calculates a normalized levenshtein distance based on levenshtein.weighted_distance
It uses the following costs for edit operations:

edit operation | cost
:------------- | :---
Insert         | 1
Remove         | 1
Replace        | 2

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.

Returns:
    float: normalized weighted levenshtein distance between s1 and s2 as a float between 0 and 100
)";

PyObject* normalized_weighted_distance(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|d", const_cast<char**>(kwlist), &py_s1, &py_s2,
                                   &score_cutoff))
  {
    return NULL;
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }
  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);

  double result = mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
        return rlevenshtein::normalized_weighted_distance(val1, val2, score_cutoff / 100);
      },
      s1_view, s2_view);

  return PyFloat_FromDouble(result * 100);
}

static PyMethodDef methods[] = {
    PY_METHOD(distance),
    PY_METHOD(normalized_distance),
    PY_METHOD(weighted_distance),
    PY_METHOD(normalized_weighted_distance),
    {NULL, NULL, 0, NULL} /* sentinel */
};

PY_INIT_MOD(levenshtein, NULL, methods)