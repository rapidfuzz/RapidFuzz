/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */
/* Copyright © 2011 Adam Cohen */

#include "fuzz.hpp"
#include "py_utils.hpp"
#include "utils.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace utils = rapidfuzz::utils;

bool use_preprocessing(PyObject* processor, bool processor_default)
{
  return processor ? PyObject_IsTrue(processor) : processor_default;
}

// TODO: make this the default when partial_ratio accepts strings of different char types
template <typename MatchingFunc>
static PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|Od", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  if (PyCallable_Check(processor)) {
    PyObject* proc_s1 = PyObject_CallFunctionObjArgs(processor, py_s2, NULL);
    if (proc_s1 == NULL) {
      return NULL;
    }

    PyObject* proc_s2 = PyObject_CallFunctionObjArgs(processor, py_s2, NULL);
    if (proc_s2 == NULL) {
      Py_DecRef(proc_s1);
      return NULL;
    }

    auto s1_view = decode_python_string(proc_s1);
    auto s2_view = decode_python_string(proc_s2);

    double result = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return MatchingFunc::call(val1, val2, score_cutoff);
        },
        s1_view, s2_view);

    Py_DecRef(proc_s1);
    Py_DecRef(proc_s2);

    return PyFloat_FromDouble(result);
  }

  auto s1_view = decode_python_string(py_s1);
  auto s2_view = decode_python_string(py_s2);

  double result;
  if (use_preprocessing(processor, processor_default)) {
    result = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return MatchingFunc::call(utils::default_process(val1), utils::default_process(val2),
                                    score_cutoff);
        },
        s1_view, s2_view);
  }
  else {
    result = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return MatchingFunc::call(val1, val2, score_cutoff);
        },
        s1_view, s2_view);
  }

  return PyFloat_FromDouble(result);
}

template <typename MatchingFunc>
static PyObject* fuzz_call_old(bool processor_default, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|Od", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  if (PyCallable_Check(processor)) {
    PyObject* py_proc_s1 = PyObject_CallFunctionObjArgs(processor, py_s1, NULL);
    if (py_proc_s1 == NULL) {
      return NULL;
    }

    auto proc_s1_view = decode_python_string(py_proc_s1);
    std::wstring proc_s1 =
        mpark::visit([](auto&& val) { return std::wstring(val.begin(), val.end()); }, proc_s1_view);
    Py_DecRef(py_proc_s1);

    PyObject* py_proc_s2 = PyObject_CallFunctionObjArgs(processor, py_s2, NULL);
    if (py_proc_s2 == NULL) {
      return NULL;
    }

    auto proc_s2_view = decode_python_string(py_proc_s2);
    std::wstring proc_s2 =
        mpark::visit([](auto&& val) { return std::wstring(val.begin(), val.end()); }, proc_s2_view);
    Py_DecRef(py_proc_s2);

    auto result = MatchingFunc::call(proc_s1, proc_s2, score_cutoff);
    return PyFloat_FromDouble(result);
  }

  auto s1_view = decode_python_string(py_s1);
  std::wstring s1 =
      mpark::visit([](auto&& val) { return std::wstring(val.begin(), val.end()); }, s1_view);

  auto s2_view = decode_python_string(py_s2);
  std::wstring s2 =
      mpark::visit([](auto&& val) { return std::wstring(val.begin(), val.end()); }, s2_view);

  double result;
  if (use_preprocessing(processor, processor_default)) {
    result =
        MatchingFunc::call(utils::default_process(s1), utils::default_process(s2), score_cutoff);
  }
  else {
    result = MatchingFunc::call(s1, s2, score_cutoff);
  }

  return PyFloat_FromDouble(result);
}

PyDoc_STRVAR(
    ratio_docstring,
    "ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
    "--\n\n"
    "calculates a simple ratio between two strings\n\n"
    "Args:\n"
    "    s1 (str): first string to compare\n"
    "    s2 (str): second string to compare\n"
    "    processor (Union[bool, Callable]): optional callable that reformats the strings. None\n"
    "        is used by default.\n"
    "    score_cutoff (float): Optional argument for a score threshold as a float between 0 and "
    "100.\n"
    "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
    "Returns:\n"
    "    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
    "Example:\n"
    "    >>> fuzz.ratio(\"this is a test\", \"this is a test!\")\n"
    "    96.55171966552734");

struct ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::ratio(std::forward<Args>(args)...);
  }
};

static PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<ratio_func>(false, args, keywds);
}

PyDoc_STRVAR(
    partial_ratio_docstring,
    "partial_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
    "--\n\n"
    "calculates the fuzz.ratio of the optimal string alignment\n\n"
    "Args:\n"
    "    s1 (str): first string to compare\n"
    "    s2 (str): second string to compare\n"
    "    processor (Union[bool, Callable]): optional callable that reformats the strings. None\n"
    "        is used by default.\n"
    "    score_cutoff (float): Optional argument for a score threshold as a float between 0 and "
    "100.\n"
    "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
    "Returns:\n"
    "    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
    "Example:\n"
    "    >>> fuzz.partial_ratio(\"this is a test\", \"this is a test!\")\n"
    "    100");

struct partial_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::partial_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call_old<partial_ratio_func>(false, args, keywds);
}

PyDoc_STRVAR(
    token_sort_ratio_docstring,
    "token_sort_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
    "--\n\n"
    "sorts the words in the strings and calculates the fuzz.ratio between them\n\n"
    "Args:\n"
    "    s1 (str): first string to compare\n"
    "    s2 (str): second string to compare\n"
    "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
    "utils.default_process\n"
    "        is used by default, which lowercases the strings and trims whitespace\n"
    "    score_cutoff (float): Optional argument for a score threshold as a float between 0 and "
    "100.\n"
    "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
    "Returns:\n"
    "    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
    "Example:\n"
    "    >>> fuzz.token_sort_ratio(\"fuzzy wuzzy was a bear\", \"wuzzy fuzzy was a bear\")\n"
    "    100.0");

struct token_sort_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::token_sort_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<token_sort_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(partial_token_sort_ratio_docstring,
             "partial_token_sort_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "sorts the words in the strings and calculates the fuzz.partial_ratio between them\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100");

struct partial_token_sort_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::partial_token_sort_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call_old<partial_token_sort_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(token_set_ratio_docstring,
             "token_set_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "Compares the words in the strings based on unique and common words between them "
             "using fuzz.ratio\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
             "Example:\n"
             "    >>> fuzz.token_sort_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")\n"
             "    83.8709716796875\n"
             "    >>> fuzz.token_set_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")\n"
             "    100.0");

struct token_set_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::token_set_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<token_set_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(partial_token_set_ratio_docstring,
             "partial_token_set_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "Compares the words in the strings based on unique and common words between them "
             "using fuzz.partial_ratio\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100");

struct partial_token_set_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::partial_token_set_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call_old<partial_token_set_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(
    token_ratio_docstring,
    "token_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
    "--\n\n"
    "Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio\n"
    "    (faster than manually executing the two functions)\n\n"
    "Args:\n"
    "    s1 (str): first string to compare\n"
    "    s2 (str): second string to compare\n"
    "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
    "utils.default_process\n"
    "        is used by default, which lowercases the strings and trims whitespace\n"
    "    score_cutoff (float): Optional argument for a score threshold as a float between 0 and "
    "100.\n"
    "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
    "Returns:\n"
    "    float: ratio between s1 and s2 as a float between 0 and 100");

struct token_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::token_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<token_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(partial_token_ratio_docstring,
             "partial_token_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "Helper method that returns the maximum of fuzz.partial_token_set_ratio and "
             "fuzz.partial_token_sort_ratio\n"
             "    (faster than manually executing the two functions)\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100");

struct partial_token_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::partial_token_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call_old<partial_token_ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(WRatio_docstring,
             "WRatio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "Calculates a weighted ratio based on the other ratio algorithms\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100");

struct WRatio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::WRatio(std::forward<Args>(args)...);
  }
};

static PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call_old<WRatio_func>(true, args, keywds);
}

PyDoc_STRVAR(QRatio_docstring,
             "QRatio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "calculates a quick ratio between two strings using fuzz.ratio\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
             "Example:\n"
             "    >>> fuzz.QRatio(\"this is a test\", \"this is a test!\")\n"
             "    96.55171966552734");

static PyObject* QRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<ratio_func>(true, args, keywds);
}

PyDoc_STRVAR(quick_lev_ratio_docstring,
             "quick_lev_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
             "--\n\n"
             "Calculates a quick estimation of fuzz.ratio by counting uncommon letters between the "
             "two sentences.\n"
             "Guaranteed to be equal or higher than fuzz.ratio.\n"
             "(internally used by fuzz.ratio when providing it with a score_cutoff to speed up the "
             "matching)\n\n"
             "Args:\n"
             "    s1 (str): first string to compare\n"
             "    s2 (str): second string to compare\n"
             "    processor (Union[bool, Callable]): optional callable that reformats the strings. "
             "utils.default_process\n"
             "        is used by default, which lowercases the strings and trims whitespace\n"
             "    score_cutoff (float): Optional argument for a score threshold as a float between "
             "0 and 100.\n"
             "        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
             "Returns:\n"
             "    float: ratio between s1 and s2 as a float between 0 and 100");

struct quick_lev_ratio_func {
  template <typename... Args>
  static double call(Args&&... args)
  {
    return rfuzz::quick_lev_ratio(std::forward<Args>(args)...);
  }
};

static PyObject* quick_lev_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<quick_lev_ratio_func>(true, args, keywds);
}

static PyMethodDef methods[] = {
    PY_METHOD(ratio),
    PY_METHOD(partial_ratio),
    PY_METHOD(token_sort_ratio),
    PY_METHOD(partial_token_sort_ratio),
    PY_METHOD(token_set_ratio),
    PY_METHOD(partial_token_set_ratio),
    PY_METHOD(token_ratio),
    PY_METHOD(partial_token_ratio),
    PY_METHOD(WRatio),
    PY_METHOD(QRatio),
    PY_METHOD(quick_lev_ratio),
    {NULL, NULL, 0, NULL} /* sentinel */
};

PY_INIT_MOD(fuzz, NULL, methods)