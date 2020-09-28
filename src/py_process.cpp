#include "fuzz.hpp"
#include "py_utils.hpp"
#include "utils.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace utils = rapidfuzz::utils;

PyObject* extractOne(PyObject* self, PyObject* args, PyObject* keywds)
{
  PyObject* py_query;
  PyObject* py_choices;
  PyObject* processor = NULL;
  PyObject* scorer = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"query", "choices", "scorer", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOd", const_cast<char**>(kwlist), &py_query,
                                   &py_choices, &scorer, &processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_query == Py_None) {
    return PyFloat_FromDouble(0);
  }

  if (PyObject_HasAttrString(py_choices, "items")) {
  }
  else {
  }

  if (PySequence_Check(processor)) {
  }

  if (!valid_str(py_query, "query")) {
    return NULL;
  }

  // if is list

  PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
  if (!choices) {
    return NULL;
  }

  std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);

  bool match_found;
  // PyObject*

  // processing missing
  auto query_view = decode_python_string(py_query);

  for (std::size_t i = 0; i < choice_count; ++i) {
    PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

    if (py_choice == Py_None) {
      continue;
    }

    if (!valid_str(py_choice, "choice")) {
      Py_DECREF(choices);
      return NULL;
    }

    auto choice_view = decode_python_string(py_choice);

    double score = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::WRatio(val1, val2, score_cutoff);
        },
        query_view, choice_view);
    /*
        float score;
        if (preprocess) {
          score = fuzz::WRatio(
            cleaned_query,
            utils::default_process(choice),
            score_cutoff);
        } else {
          score = fuzz::WRatio(
            cleaned_query,
            std::wstring_view(choice, wcslen(choice)),
            score_cutoff);
        }*/

    if (score >= score_cutoff) {
      // increase the value by a small step so it might be able to exit early
      score_cutoff = score + (float)0.00001;
      match_found = true;
      result_choice = choice;
    }
  }

  Py_DECREF(choices);

  if (!match_found) {
    Py_RETURN_NONE;
  }

  if (score_cutoff > 100) {
    score_cutoff = 100;
  }
  return Py_BuildValue("(ud)", result_choice, score_cutoff);
}
