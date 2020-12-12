/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "py_string_metric.hpp"
#include "utils.hpp"
#include "py_process.hpp"
#include "py_utils.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace rutils = rapidfuzz::utils;

static inline void free_owner_list(const std::vector<PyObject*>& owner_list)
{
  for (const auto owned : owner_list) {
    Py_DecRef(owned);
  }
}

std::unique_ptr<CachedScorer> get_matching_instance(PyObject* scorer)
{
  if (scorer) {
    if (PyCFunction_Check(scorer)) {
      auto scorer_func = PyCFunction_GetFunction(scorer);
      if (scorer_func == PY_FUNC_CAST(ratio)) {
        return std::unique_ptr<CachedRatio>(new CachedRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(partial_ratio)) {
        return std::unique_ptr<CachedPartialRatio>(new CachedPartialRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(token_sort_ratio)) {
        return std::unique_ptr<CachedTokenSortRatio>(new CachedTokenSortRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(token_set_ratio)) {
        return std::unique_ptr<CachedTokenSetRatio>(new CachedTokenSetRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_sort_ratio)) {
        return std::unique_ptr<CachedPartialTokenSortRatio>(new CachedPartialTokenSortRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_set_ratio)) {
        return std::unique_ptr<CachedPartialTokenSetRatio>(new CachedPartialTokenSetRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(token_ratio)) {
        return std::unique_ptr<CachedTokenRatio>(new CachedTokenRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_ratio)) {
        return std::unique_ptr<CachedPartialTokenRatio>(new CachedPartialTokenRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(WRatio)) {
        return std::unique_ptr<CachedWRatio>(new CachedWRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(QRatio)) {
        return std::unique_ptr<CachedQRatio>(new CachedQRatio());
      }
      else if (scorer_func == PY_FUNC_CAST(normalized_letter_frequency)) {
        return std::unique_ptr<CachedNormalizedLetterFrequency>(new CachedNormalizedLetterFrequency());
      }
    }
    /* call python function */
    return nullptr;
    /* default is fuzz.WRatio */
  }
  else {
    return std::unique_ptr<CachedWRatio>(new CachedWRatio());
  }
}

// C++11 does not support generic lambdas
struct EncodePythonStringVisitor {
  template <typename Sentence>
  PyObject* operator()(Sentence&& s) const
  {
    return encode_python_string(s);
  }
};

static PyObject* py_extractOne(PyObject* py_query, PyObject* py_choices, PyObject* scorer,
                               std::unique_ptr<Processor> processor, double score_cutoff)
{
  PyObject* result_choice = NULL;
  PyObject* choice_key = NULL;
  Py_ssize_t result_index = -1;
  std::vector<PyObject*> outer_owner_list;

  bool is_dict = false;

  PyObject* py_score_cutoff = PyFloat_FromDouble(score_cutoff);
  if (!py_score_cutoff) {
    return NULL;
  }

  try {
    auto query = processor->call(py_query, "query");
    py_query = mpark::visit(EncodePythonStringVisitor(), query.value);
    if (!py_query) {
      throw std::invalid_argument("");
    }
  
    outer_owner_list.push_back(py_query);

    /* dict like container */
    if (PyObject_HasAttrString(py_choices, "items")) {
      is_dict = true;
      py_choices = PyObject_CallMethod(py_choices, "items", NULL);
      if (!py_choices) {
        throw std::invalid_argument("");
      }
      outer_owner_list.push_back(py_choices);
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
      throw std::invalid_argument("");
    }
    outer_owner_list.push_back(choices);

    Py_ssize_t choice_count = PySequence_Fast_GET_SIZE(choices);

    for (Py_ssize_t i = 0; i < choice_count; ++i) {
      PyObject* py_choice = NULL;
      PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, i);

      if (is_dict) {
        if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
          throw std::invalid_argument("");
        }
      }

      if (py_match_choice == Py_None) {
        continue;
      }

      auto choice = processor->call(py_match_choice, "choice");
      PyObject* py_proc_choice = mpark::visit(EncodePythonStringVisitor(), choice.value);

      if (!py_proc_choice) {
        throw std::invalid_argument("");
      }

      PyObject* score =
          PyObject_CallFunction(scorer, "OOO", py_query, py_proc_choice, py_score_cutoff);

      Py_DecRef(py_proc_choice);

      if (!score) {
        throw std::invalid_argument("");
      }

      int comp = PyObject_RichCompareBool(score, py_score_cutoff, Py_GE);
      if (comp == 1) {
        Py_DecRef(py_score_cutoff);
        py_score_cutoff = score;
        result_choice = py_match_choice;
        choice_key = py_choice;
        result_index = i;
      }
      else if (comp == 0) {
        Py_DecRef(score);
      }
      else if (comp == -1) {
        Py_DecRef(score);
        throw std::invalid_argument("");
      }
    }

  } catch(std::invalid_argument& e) {
    // todo replace
    free_owner_list(outer_owner_list);
    Py_DecRef(py_score_cutoff);
    return NULL;
  }  

  if (result_index == -1) {
    free_owner_list(outer_owner_list);
    Py_DecRef(py_score_cutoff);
    Py_RETURN_NONE;
  }

  PyObject* result = is_dict ? Py_BuildValue("(OOO)", result_choice, py_score_cutoff, choice_key)
                             : Py_BuildValue("(OOn)", result_choice, py_score_cutoff, result_index);

  free_owner_list(outer_owner_list);
  Py_DecRef(py_score_cutoff);
  return result;
}

PyObject* extractOne(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* result_choice = NULL;
  PyObject* choice_key = NULL;
  double result_score;
  Py_ssize_t result_index = -1;
  std::vector<PyObject*> outer_owner_list;
  python_string query;
  bool is_dict = false;

  PyObject* py_query;
  PyObject* py_choices;
  PyObject* py_processor = NULL;
  PyObject* py_scorer = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"query", "choices", "scorer", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOd", const_cast<char**>(kwlist), &py_query,
                                   &py_choices, &py_scorer, &py_processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_query == Py_None) {
    return PyFloat_FromDouble(0);
  }

  auto scorer = get_matching_instance(py_scorer);
  auto processor = get_processor(py_processor, true);

  if (!scorer) {
    // todo this is mostly code duplication
    return py_extractOne(py_query, py_choices, py_scorer, std::move(processor), score_cutoff);
  }

  try {
    auto query = processor->call(py_query, "query");
    scorer->str1_set(query.value);

    /* dict like container */
    if (PyObject_HasAttrString(py_choices, "items")) {
      is_dict = true;
      py_choices = PyObject_CallMethod(py_choices, "items", NULL);
      if (!py_choices) {
        throw std::invalid_argument("");
      }
      outer_owner_list.push_back(py_choices);
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
      throw std::invalid_argument("");
    }
    outer_owner_list.push_back(choices);

    Py_ssize_t choice_count = PySequence_Fast_GET_SIZE(choices);

    for (Py_ssize_t i = 0; i < choice_count; ++i) {
      PyObject* py_choice = NULL;
      PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, i);

      if (is_dict) {
        if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
          throw std::invalid_argument("");
        }
      }

      if (py_match_choice == Py_None) {
        continue;
      }

      auto choice = processor->call(py_match_choice, "choice");
      scorer->str2_set(choice.value);

      double score = scorer->call(score_cutoff);

      if (score >= score_cutoff) {
        // increase the value by a small step so it might be able to exit early
        score_cutoff = score + (float)0.00001;
        result_score = score;
        result_choice = py_match_choice;
        choice_key = py_choice;
        result_index = i;

        if (score_cutoff > 100) {
          break;
        }
      }
    }
  } catch(std::invalid_argument& e) {
    free_owner_list(outer_owner_list);
    return NULL;
  }

  if (result_index == -1) {
    free_owner_list(outer_owner_list);
    Py_RETURN_NONE;
  }

  PyObject* result = is_dict ? Py_BuildValue("(OdO)", result_choice, result_score, choice_key)
                             : Py_BuildValue("(Odn)", result_choice, result_score, result_index);

  free_owner_list(outer_owner_list);
  return result;
}
