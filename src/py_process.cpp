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

//template <typename Scorer>
template <template<typename> class Scorer>
struct GenericScorerAllocVisitor {
  template <typename Sentence>
  std::unique_ptr<CachedScorer> operator()(Sentence&& query) {
    return std::unique_ptr<CachedScorer>(
      new GenericCachedScorer<Scorer, Sentence>(std::forward<Sentence>(query))
    );
  }
};

std::unique_ptr<CachedScorer> get_matching_instance(PyObject* scorer, const python_string& query)
{
  if (scorer) {
    if (PyCFunction_Check(scorer)) {
      auto scorer_func = PyCFunction_GetFunction(scorer);
      if (scorer_func == PY_FUNC_CAST(ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(partial_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedPartialRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(token_sort_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedTokenSortRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(token_set_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedTokenSetRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_sort_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedPartialTokenSortRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_set_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedPartialTokenSetRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(token_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedTokenRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(partial_token_ratio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedPartialTokenRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(WRatio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedWRatio>(), query);
      }
      else if (scorer_func == PY_FUNC_CAST(QRatio)) {
        return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedQRatio>(), query);
      }
    }
    /* call python function */
    return nullptr;
    /* default is fuzz.WRatio */
  }
  else {
    return mpark::visit(GenericScorerAllocVisitor<rfuzz::CachedWRatio>(), query);
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

  PyObject* kwargs =  PyDict_New();
  if (!kwargs) {
    Py_DecRef(py_score_cutoff);
    return NULL;
  }
  outer_owner_list.push_back(kwargs);

  PyDict_SetItemString(kwargs, "processor", Py_None);
  PyDict_SetItemString(kwargs, "score_cutoff", py_score_cutoff);
  Py_DecRef(py_score_cutoff);

  PyObject* args = PyTuple_New(2);
  if (!args) {
    free_owner_list(outer_owner_list);
    return NULL;
  }
  outer_owner_list.push_back(args);

  try {
    auto query = processor->call(py_query, "query");
    py_query = mpark::visit(EncodePythonStringVisitor(), query.value);
    if (!py_query) {
      throw std::invalid_argument("");
    }

    PyTuple_SET_ITEM(args, 0, py_query);

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

      PyTuple_SetItem(args, 1, py_proc_choice);

      PyObject* score = PyObject_Call(scorer, args, kwargs);

      if (!score) {
        throw std::invalid_argument("");
      }

      int comp = PyObject_RichCompareBool(score, py_score_cutoff, Py_GT);
      if (comp == 1) {
        py_score_cutoff = score;
        PyDict_SetItemString(kwargs, "score_cutoff", score);
        result_choice = py_match_choice;
        choice_key = py_choice;
        result_index = i;
      }
      else if (comp == -1) {
        Py_DecRef(score);
        throw std::invalid_argument("");
      }
      Py_DecRef(score);
    }

  } catch(std::invalid_argument& e) {
    // todo replace
    free_owner_list(outer_owner_list);
    return NULL;
  }  

  if (result_index == -1) {
    free_owner_list(outer_owner_list);
    Py_RETURN_NONE;
  }

  PyObject* result = is_dict ? Py_BuildValue("(OOO)", result_choice, py_score_cutoff, choice_key)
                             : Py_BuildValue("(OOn)", result_choice, py_score_cutoff, result_index);

  free_owner_list(outer_owner_list);
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

  try {
    auto processor = get_processor(py_processor, true);
    //todo do not run twice
    auto query = processor->call(py_query, "query");

    auto scorer = get_matching_instance(py_scorer, query.value);

    if (!scorer) {
      // todo this is mostly code duplication
      return py_extractOne(py_query, py_choices, py_scorer, std::move(processor), score_cutoff);
    }

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

      //todo changed back
      double score = scorer->ratio(choice.value, 0/*score_cutoff*/);

      if (score >= score_cutoff) {
        // increase the value by a small step so it might be able to exit early
        score_cutoff = score + (float)0.00001;
        result_score = score;
        result_choice = py_match_choice;
        choice_key = py_choice;
        result_index = i;

        /*if (score_cutoff > 100) {
          break;
        }*/
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
