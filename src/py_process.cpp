/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "py_string_metric.hpp"
#include "utils.hpp"
#include "py_process.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace rutils = rapidfuzz::utils;
namespace string_metric = rapidfuzz::string_metric;


static inline double calc_similarity(
  PyObject* py_choice, PyObject* py_processor, processor_func processor,
  CachedScorer* scorer, double score_cutoff)
{
  double score = 0;

  switch(processor.index()) {
  case 0: /* No Processor */
  {
    if (!valid_str(py_choice, "choice")) throw std::invalid_argument("");
    auto choice = decode_python_string(py_choice);
    score = scorer->ratio(choice, score_cutoff);
    break;
  }
  case 1: /* Python processor */
  {
    auto choice = mpark::get<1>(processor)(py_processor, py_choice, "choice");
    score = scorer->ratio(choice.value, score_cutoff);
    break;
  }
  case 2: /* C++ processor */
  {
    auto choice = mpark::get<2>(processor)(py_choice, "choice");
    score = scorer->ratio(choice, score_cutoff);
    break;
  }
  }

  return score;
}

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

static std::unique_ptr<CachedScorer> get_matching_instance(PyObject* scorer, const python_string& query)
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
      else if (scorer_func == PY_FUNC_CAST(normalized_hamming)) {
        return mpark::visit(GenericScorerAllocVisitor<string_metric::CachedNormalizedHamming>(), query);
      }
    }
    /* call python function */
    return nullptr;
  }
  else {
    /* default is fuzz.WRatio */
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
                               PyObject* py_processor, processor_func processor, double score_cutoff)
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
    auto query = preprocess(py_query, py_processor, processor, "query");
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

      auto choice = preprocess(py_match_choice, py_processor, processor, "choice");

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
    Py_RETURN_NONE;
  }

  try {
    auto processor = get_processor(py_processor, true);
    //todo do not run twice for python scorer
    auto query = preprocess(py_query, py_processor, processor, "query");

    auto scorer = get_matching_instance(py_scorer, query.value);

    if (!scorer) {
      // todo this is mostly code duplication
      return py_extractOne(py_query, py_choices, py_scorer, py_processor, processor, score_cutoff);
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

      double score = calc_similarity(py_match_choice, py_processor, processor, scorer.get(), score_cutoff);

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

struct ExtractComp
{
    template<class T>
    bool operator()(T const &a, T const &b) const {
        if (a.first > b.first) {
            return true;
        } else if (a.first < b.first) {
            return false;
        } else {
            return a.second < b.second;
        }
    }
};

static inline PyObject* extract_resultsToList(
  const std::vector<std::pair<double, Py_ssize_t>>& results,
  Py_ssize_t limit,
  bool is_dict,
  PyObject* choices)
{
  PyObject* result_list = PyList_New(limit);
  if (result_list == NULL) {
    goto Error;
  }

  for (Py_ssize_t i = 0; i < limit; ++i) {
    double score = results[i].first;
    Py_ssize_t index = results[i].second;
    PyObject* result_tuple = NULL;

    if (is_dict) {
      PyObject* py_choice = NULL;
      PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, index);
      PyObject* py_score = NULL;

      if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
        goto Error;
      }

      py_score = PyFloat_FromDouble(score);
      if (py_score == NULL) {
        goto Error;
      }

      result_tuple = PyTuple_Pack(3, py_match_choice, py_score, py_choice);
      Py_DecRef(py_score);
    } else {
      PyObject* py_match_choice = PySequence_Fast_GET_ITEM(choices, index);
      PyObject* py_score = NULL;
      PyObject* py_choice = NULL;

      py_score = PyFloat_FromDouble(score);
      if (py_score == NULL) {
        goto Error;
      }

      py_choice = PyLong_FromSsize_t(index);
      if (py_choice == NULL) {
        Py_DecRef(py_score);
        goto Error;
      }

      result_tuple = PyTuple_Pack(3, py_match_choice, py_score, py_choice);
      Py_DecRef(py_score);
      Py_DecRef(py_choice);
    }

    if (result_tuple == NULL) {
      goto Error;
    }

    PyList_SET_ITEM(result_list, i, result_tuple);
  }

  return result_list;
Error:
  Py_DecRef(result_list);
  return NULL;
}

PyObject* extract(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  std::vector<PyObject*> outer_owner_list;
  std::vector<std::pair<double, Py_ssize_t>> results;
  python_string query;
  bool is_dict = false;

  PyObject* py_query;
  PyObject* py_choices;
  PyObject* choices;
  PyObject* py_processor = NULL;
  PyObject* py_scorer = NULL;
  PyObject* py_limit = NULL;
  Py_ssize_t limit = 5;
  double score_cutoff = 0;
  static const char* kwlist[] = {"query", "choices", "scorer", "processor", "limit", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOOd", const_cast<char**>(kwlist), &py_query,
                                   &py_choices, &py_scorer, &py_processor, &py_limit, &score_cutoff))
  {
    return NULL;
  }

  if (py_query == Py_None) {
    // todo
    Py_RETURN_NONE;
  }

  if (py_limit != NULL) {
    if (py_limit == Py_None) {
      limit = -1;
    } else {
      if (PyLong_Check(py_limit)) {
        limit = PyLong_AsSsize_t(py_limit);
        if (limit == -1 && PyErr_Occurred()) {
          return NULL;
        }
      }
#if PY_VERSION_HEX < PYTHON_VERSION(3, 0, 0)
      else if (PyInt_Check(py_limit)) {
        limit = PyInt_AsSsize_t(py_limit);
        if (limit == -1 && PyErr_Occurred()) {
          return NULL;
        }
      }
#endif
      else {
        // todo exception
        PyErr_SetString(PyExc_TypeError, "limit has to be a Integer or None");
        return NULL;
      }
    }
  }

  try {
    auto processor = get_processor(py_processor, true);
    auto query = preprocess(py_query, py_processor, processor, "query");

    auto scorer = get_matching_instance(py_scorer, query.value);

    if (!scorer) {
      // use Python implementation, since we would have to call the scorer through python anyways
      PyErr_SetString(PyExc_TypeError, "The C++ implementation only supports scorers implemented in C++");
      return NULL;
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

    choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
      throw std::invalid_argument("");
    }
    outer_owner_list.push_back(choices);

    Py_ssize_t choice_count = PySequence_Fast_GET_SIZE(choices);
    results.reserve(choice_count);

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

      double score = calc_similarity(py_match_choice, py_processor, processor, scorer.get(), score_cutoff);

      if (score >= score_cutoff) {
        results.emplace_back(score, i);
      }

    }
  } catch(std::invalid_argument& e) {
    free_owner_list(outer_owner_list);
    return NULL;
  }

  if (limit < 0 || limit >= results.size()) {
    limit = results.size();
  }

  if (limit == results.size()) {
    std::sort(results.begin(), results.end(), ExtractComp());
  } else {
    std::partial_sort(results.begin(), results.begin() + limit, results.end(), ExtractComp());
  }

  PyObject* result_list = extract_resultsToList(results, limit, is_dict, choices);

  free_owner_list(outer_owner_list);
  return result_list;
}



PyObject* extract_iter_new(PyTypeObject *type, PyObject *args, PyObject *keywds)
{
  PyObject* py_query;
  PyObject* py_choices;
  PyObject* py_processor = NULL;
  PyObject* py_scorer = NULL;
  PyObject* py_score_cutoff = NULL;
  static const char* kwlist[] = {"query", "choices", "scorer", "processor", "score_cutoff", NULL};


  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOO", const_cast<char**>(kwlist), &py_query,
                                   &py_choices, &py_scorer, &py_processor, &py_score_cutoff))
  {
    return NULL;
  }

  /* Create a new ExtractIterState and initialize its state */
  ExtractIterState *state = (ExtractIterState *)type->tp_alloc(type, 0);
  if (!state) {
    return NULL;
  }

  // store choices
  if (PyObject_HasAttrString(py_choices, "items")) {
    state->is_dict = true;
    state->choicesObj = PyObject_CallMethod(py_choices, "items", NULL);
    if (!state->choicesObj) {
      goto Error;
    }
  } else {
    state->is_dict = false;
    Py_INCREF(py_choices);
    state->choicesObj = py_choices;
  }
  state->choices = PySequence_Fast(state->choicesObj, "Choices must be a sequence of strings");
  if (!state->choices) {
    goto Error;
  }

  state->choice_count = PySequence_Fast_GET_SIZE(state->choices);
  state->choice_index = 0;

  // store processor
  Py_XINCREF(py_processor);
  state->processorObj = py_processor;
  state->processor = get_processor(py_processor, true);

  // store query
  Py_INCREF(py_query);
  state->queryObj = py_query;
  try {
    state->query = preprocess(py_query, state->processorObj, state->processor, "query");
  } catch(std::invalid_argument& e) {
    goto Error;
  }

  // store and init scorer
  Py_XINCREF(py_scorer);
  state->scorerObj = py_scorer;
  state->scorer = get_matching_instance(py_scorer, state->query.value);

  if(py_score_cutoff) {
    if (state->scorer) {
      state->score_cutoff = PyFloat_AsDouble(py_score_cutoff);
    } else {
      Py_INCREF(py_scorer);
      state->scoreCutoffObj = py_score_cutoff;
    }
  } else {
    if (state->scorer) {
      state->score_cutoff = 0;
    } else {
      state->scoreCutoffObj = PyFloat_FromDouble(0);
    }
  }


  // scorer is a python function
  if (!state->scorer) {
    PyObject* py_proc_query = NULL;

    state->kwargsObj = PyDict_New();
    if (!state->kwargsObj) {
      goto Error;
    }

    PyDict_SetItemString(state->kwargsObj, "processor", Py_None);
    PyDict_SetItemString(state->kwargsObj, "score_cutoff", state->scoreCutoffObj);

    state->argsObj = PyTuple_New(2);
    if (!state->argsObj) {
      goto Error;
    }

    py_proc_query = mpark::visit(EncodePythonStringVisitor(), state->query.value);
    if (!py_proc_query) {
      goto Error;
    }

    PyTuple_SET_ITEM(state->argsObj, 0, py_proc_query);
  }


  return (PyObject *)state;

Error:
  Py_XDECREF(state->choicesObj);
  Py_XDECREF(state->choices);
  Py_XDECREF(state->processorObj);
  Py_XDECREF(state->queryObj);
  Py_XDECREF(state->scorerObj);
  Py_XDECREF(state->argsObj);
  Py_XDECREF(state->kwargsObj);
  Py_XDECREF(state->scoreCutoffObj);

  Py_TYPE(state)->tp_free(state);
  return NULL;
}


void extract_iter_dealloc(ExtractIterState *state)
{
    /* We need XDECREF here because when the generator is exhausted,
     * rgstate->sequence is cleared with Py_CLEAR which sets it to NULL.
    */

  Py_XDECREF(state->choicesObj);
  Py_XDECREF(state->choices);
  Py_XDECREF(state->processorObj);
  Py_XDECREF(state->queryObj);
  Py_XDECREF(state->scorerObj);
  Py_XDECREF(state->argsObj);
  Py_XDECREF(state->kwargsObj);
  Py_XDECREF(state->scoreCutoffObj);

  Py_TYPE(state)->tp_free(state);
}


PyObject* extract_iter_next(ExtractIterState *state)
{
  /* seq_index < 0 means that the generator is exhausted.
   * Returning NULL in this case is enough. The next() builtin will raise the
   * StopIteration error for us.
  */
  if (state->choice_index < state->choice_count) {
    PyObject* py_choice = NULL;
    PyObject* py_match_choice = PySequence_Fast_GET_ITEM(state->choices, state->choice_index);

    if (state->is_dict) {
      if (!PyArg_ParseTuple(py_match_choice, "OO", &py_choice, &py_match_choice)) {
        return NULL;
      }
    }

    PyObject* result;
    if (py_match_choice != Py_None) {
      try {
        auto choice = preprocess(py_match_choice, state->processorObj, state->processor, "choice");

        // built in scorer of rapidfuzz
        if (state->scorer) {
          double score = state->scorer->ratio(choice.value, state->score_cutoff);
          if (score < state->score_cutoff) {
            state->choice_index += 1;
            return extract_iter_next(state);
          }
          result = state->is_dict ? Py_BuildValue("(OdO)", py_match_choice, score, py_choice)
                                  : Py_BuildValue("(Odn)", py_match_choice, score, state->choice_index);
        // custom scorer that has to be called through Python
        } else {
          PyObject* py_proc_choice = mpark::visit(EncodePythonStringVisitor(), choice.value);
          if (!py_proc_choice) {
            return NULL;
          }
          PyTuple_SetItem(state->argsObj, 1, py_proc_choice);
          PyObject* score = PyObject_Call(state->scorerObj, state->argsObj, state->kwargsObj);
          if (!score) {
            return NULL;
          }

          int comp = PyObject_RichCompareBool(score, state->scoreCutoffObj, Py_LT);
          // lower than
          if (comp == 1) {
            state->choice_index += 1;
            return extract_iter_next(state);
          // error in comparision
          } else if (comp == -1) {
            Py_DecRef(score);
            return NULL;
          }

          result = state->is_dict ? Py_BuildValue("(OOO)", py_match_choice, score, py_choice)
                                  : Py_BuildValue("(OOn)", py_match_choice, score, state->choice_index);
          Py_DecRef(score);
        }
      } catch(std::invalid_argument& e) {
        const char* msg = e.what();
        if (msg[0]) {
          PyErr_SetString(PyExc_ValueError, msg);
        }
        return NULL;
      }
    } else {
      result = state->is_dict ? Py_BuildValue("(OdO)", py_match_choice, 0.0, py_choice)
                              : Py_BuildValue("(Odn)", py_match_choice, 0.0, state->choice_index);
    }

    state->choice_index += 1;
    return result;
  }

  /* The reference to the sequence is cleared in the first generator call
   * after its exhaustion (after the call that returned the last element).
   * Py_CLEAR will be harmless for subsequent calls since it's idempotent
   * on NULL.
  */
  Py_CLEAR(state->choices);
  return NULL;
}