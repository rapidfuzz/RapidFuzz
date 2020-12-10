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

static inline bool use_preprocessing(PyObject* processor, bool processor_default)
{
  return processor ? PyObject_IsTrue(processor) != 0 : processor_default;
}

static inline bool non_default_process(PyObject* processor)
{
  if (processor) {
    if (PyCFunction_Check(processor)) {
      if (PyCFunction_GetFunction(processor) == PY_FUNC_CAST(default_process)) {
        return false;
      }
    }
  }

  return PyCallable_Check(processor);
}

static inline void free_owner_list(const std::vector<PyObject*>& owner_list)
{
  for (const auto owned : owner_list) {
    Py_DecRef(owned);
  }
}

template <typename Sentence>
static inline python_string default_process_string(Sentence&& str)
{
  return rutils::default_process(std::forward<Sentence>(str));
}

// C++11 does not support generic lambdas
struct DefaultProcessVisitor {
  template <typename Sentence>
  python_string operator()(Sentence&& s) const
  {
    return default_process_string(s);
  }
};

static inline bool process_string(PyObject* py_str, const char* name, PyObject* processor,
                                  bool processor_default, python_string& proc_str,
                                  std::vector<PyObject*>& owner_list)
{
  if (non_default_process(processor)) {
    PyObject* proc_py_str = PyObject_CallFunctionObjArgs(processor, py_str, NULL);
    if ((proc_py_str == NULL) || (!valid_str(proc_py_str, name))) {
      return false;
    }

    owner_list.push_back(proc_py_str);
    proc_str = decode_python_string(proc_py_str);
    return true;
  }

  if (!valid_str(py_str, name)) {
    return false;
  }

  if (use_preprocessing(processor, processor_default)) {
    proc_str = mpark::visit(DefaultProcessVisitor(), decode_python_string(py_str));
  }
  else {
    proc_str = decode_python_string(py_str);
  }

  return true;
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
      else if (scorer_func == PY_FUNC_CAST(quick_lev_ratio)) {
        return std::unique_ptr<CachedQuickLevRatio>(new CachedQuickLevRatio());
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
                               PyObject* processor, double score_cutoff)
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

  python_string query;
  if (!process_string(py_query, "query", processor, true, query, outer_owner_list)) {
    Py_DecRef(py_score_cutoff);
    return NULL;
  }

  py_query = mpark::visit(EncodePythonStringVisitor(), query);

  if (!py_query) {
    Py_DecRef(py_score_cutoff);
    free_owner_list(outer_owner_list);
    return NULL;
  }
  outer_owner_list.push_back(py_query);

  /* dict like container */
  if (PyObject_HasAttrString(py_choices, "items")) {
    is_dict = true;
    py_choices = PyObject_CallMethod(py_choices, "items", NULL);
    if (!py_choices) {
      free_owner_list(outer_owner_list);
      return NULL;
    }
    outer_owner_list.push_back(py_choices);
  }

  PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
  if (!choices) {
    Py_DecRef(py_score_cutoff);
    free_owner_list(outer_owner_list);
    return NULL;
  }
  outer_owner_list.push_back(choices);

  Py_ssize_t choice_count = PySequence_Fast_GET_SIZE(choices);

  for (Py_ssize_t i = 0; i < choice_count; ++i) {
    PyObject* py_choice = NULL;
    PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, i);

    if (is_dict) {
      if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
        Py_DecRef(py_score_cutoff);
        free_owner_list(outer_owner_list);
        return NULL;
      }
    }

    if (py_match_choice == Py_None) {
      continue;
    }

    std::vector<PyObject*> inner_owner_list;
    python_string choice;

    if (!process_string(py_match_choice, "choice", processor, true, choice, inner_owner_list)) {
      Py_DecRef(py_score_cutoff);
      free_owner_list(outer_owner_list);
      return NULL;
    }

    PyObject* py_proc_choice = mpark::visit(EncodePythonStringVisitor(), choice);

    if (!py_proc_choice) {
      Py_DecRef(py_score_cutoff);
      free_owner_list(outer_owner_list);
      return NULL;
    }
    inner_owner_list.push_back(py_proc_choice);

    PyObject* score =
        PyObject_CallFunction(scorer, "OOO", py_query, py_proc_choice, py_score_cutoff);

    if (!score) {
      Py_DecRef(py_score_cutoff);
      free_owner_list(outer_owner_list);
      free_owner_list(inner_owner_list);
      return NULL;
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
      Py_DecRef(py_score_cutoff);
      Py_DecRef(score);
      free_owner_list(outer_owner_list);
      free_owner_list(inner_owner_list);
      return NULL;
    }
    free_owner_list(inner_owner_list);
  }

  if (result_index != -1) {
    free_owner_list(outer_owner_list);
    Py_DecRef(py_score_cutoff);
    Py_RETURN_NONE;
  }

  if (score_cutoff > 100) {
    score_cutoff = 100;
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
  PyObject* processor = NULL;
  PyObject* py_scorer = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"query", "choices", "scorer", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOd", const_cast<char**>(kwlist), &py_query,
                                   &py_choices, &py_scorer, &processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_query == Py_None) {
    return PyFloat_FromDouble(0);
  }

  auto scorer = get_matching_instance(py_scorer);
  if (!scorer) {
    // todo this is mostly code duplication
    return py_extractOne(py_query, py_choices, py_scorer, processor, score_cutoff);
  }

  if (!process_string(py_query, "query", processor, true, query, outer_owner_list)) {
    return NULL;
  }

  scorer->str1_set(query);

  /* dict like container */
  if (PyObject_HasAttrString(py_choices, "items")) {
    is_dict = true;
    py_choices = PyObject_CallMethod(py_choices, "items", NULL);
    if (!py_choices) {
      free_owner_list(outer_owner_list);
      return NULL;
    }
    outer_owner_list.push_back(py_choices);
  }

  PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
  if (!choices) {
    free_owner_list(outer_owner_list);
    return NULL;
  }
  outer_owner_list.push_back(choices);

  Py_ssize_t choice_count = PySequence_Fast_GET_SIZE(choices);

  for (Py_ssize_t i = 0; i < choice_count; ++i) {
    PyObject* py_choice = NULL;
    PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, i);

    if (is_dict) {
      if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
        free_owner_list(outer_owner_list);
        return NULL;
      }
    }

    if (py_match_choice == Py_None) {
      continue;
    }

    std::vector<PyObject*> inner_owner_list;
    python_string choice;

    if (!process_string(py_match_choice, "choice", processor, true, choice, inner_owner_list)) {
      free_owner_list(outer_owner_list);
      return NULL;
    }

    scorer->str2_set(choice);
    double score = scorer->call(score_cutoff);

    if (score >= score_cutoff) {
      // increase the value by a small step so it might be able to exit early
      score_cutoff = score + (float)0.00001;
      result_score = score;
      result_choice = py_match_choice;
      choice_key = py_choice;
      result_index = i;

      if (score_cutoff > 100) {
        free_owner_list(inner_owner_list);
        break;
      }
    }
    free_owner_list(inner_owner_list);
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
