/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */
/* Copyright © 2011 Adam Cohen */

#include "fuzz.hpp"
#include "py_utils.hpp"
#include "utils.hpp"
#include <string>

namespace rfuzz = rapidfuzz::fuzz;
namespace rutils = rapidfuzz::utils;

static PyObject* default_process(PyObject*, PyObject*, PyObject*);


static inline bool use_preprocessing(PyObject* processor, bool processor_default)
{
  return processor ? PyObject_IsTrue(processor) != 0 : processor_default;
}

static inline bool non_default_process(PyObject* processor)
{
  if (processor) {
    if (PyCFunction_Check(processor)) {
        if (PyCFunction_GetFunction(processor) == (PyCFunction)(void (*)(void))default_process) {
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

template<typename Sentence>
static inline python_string default_process_string(Sentence&& str)
{
  return rutils::default_process(std::forward<Sentence>(str));
}

static inline bool process_string(
  PyObject* py_str, PyObject* processor, bool processor_default,
  python_string& proc_str, std::vector<PyObject*>& owner_list)
{
  if (non_default_process(processor)) {
    PyObject* proc_py_str = PyObject_CallFunctionObjArgs(processor, py_str, NULL);
    if (proc_py_str == NULL) {
      return false;
    }

    owner_list.push_back(proc_py_str);
    proc_str = decode_python_string(proc_py_str);
  } else if (use_preprocessing(processor, processor_default)) {
    proc_str = mpark::visit(
        [](auto&& val1) { return default_process_string(val1);},
        decode_python_string(py_str));
  } else {
    proc_str = decode_python_string(py_str);
  }

  return true;
}

template <typename MatchingFunc>
static PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
{
  std::vector<PyObject*> owner_list;
  python_string proc_s1;
  python_string proc_s2;

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

  if (!process_string(py_s1, processor, processor_default, proc_s1, owner_list)) {
    return NULL;
  }

  if (!process_string(py_s2, processor, processor_default, proc_s2, owner_list)) {
    free_owner_list(owner_list);
    return NULL;
  }

  double result = mpark::visit(
        [score_cutoff](auto&& val1, auto&& val2) {
          return MatchingFunc::call(val1, val2, score_cutoff);
        },
        proc_s1, proc_s2);

  free_owner_list(owner_list);
  return PyFloat_FromDouble(result);
}

#define FUZZ_FUNC(name, process_default, docstring)                         \
PyDoc_STRVAR(name##_docstring, docstring);                                  \
\
struct name##_func {                                                        \
  template <typename... Args>                                               \
  static double call(Args&&... args)                                        \
  {                                                                         \
    return rfuzz::name(std::forward<Args>(args)...);                        \
  }                                                                         \
};                                                                          \
\
static PyObject* name(PyObject* /*self*/, PyObject* args, PyObject* keywds) \
{                                                                           \
  return fuzz_call<name##_func>(process_default, args, keywds);             \
}

FUZZ_FUNC(
  ratio, false,
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
  "    96.55171966552734"
)

FUZZ_FUNC(
  partial_ratio, false,
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
  "    100"
)


FUZZ_FUNC(
  token_sort_ratio, true,
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
  "    100.0"
)

FUZZ_FUNC(
  partial_token_sort_ratio, true,
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
  "    float: ratio between s1 and s2 as a float between 0 and 100"
)

FUZZ_FUNC(
  token_set_ratio, true,
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
  "    100.0"
)

FUZZ_FUNC(
  partial_token_set_ratio, true,
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
  "    float: ratio between s1 and s2 as a float between 0 and 100"
)

FUZZ_FUNC(
  token_ratio, true,
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
    "    float: ratio between s1 and s2 as a float between 0 and 100"
)

FUZZ_FUNC(
  partial_token_ratio, true,
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
  "    float: ratio between s1 and s2 as a float between 0 and 100"
)

FUZZ_FUNC(
  WRatio, true,
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
  "    float: ratio between s1 and s2 as a float between 0 and 100"
)

FUZZ_FUNC(
  QRatio, true,
  "QRatio($module, s1, s2, processor = False, score_cutoff = 0)\n"
  "--\n\n"
  "Calculates a quick ratio between two strings using fuzz.ratio\n\n"
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
  "    96.55171966552734"
)

FUZZ_FUNC(
  quick_lev_ratio, true,
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
  "    float: ratio between s1 and s2 as a float between 0 and 100"
)



constexpr const char* default_process_docstring = R"()";

static PyObject* default_process(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  PyObject* py_sentence;
  static const char* kwlist[] = {"sentence", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", const_cast<char**>(kwlist), &py_sentence)) {
    return NULL;
  }

  if (!valid_str(py_sentence, "sentence")) {
    return NULL;
  }

  auto sentence_view = decode_python_string(py_sentence);
  PyObject* processed = mpark::visit(
        [](auto&& val1) {
          return encode_python_string(rutils::default_process(val1));},
        sentence_view);
  
  return processed;
}

static PyMethodDef methods[] = {
/* utils */
    PY_METHOD(default_process),

/* fuzz */
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
/* process */
/* sentinel */
    {NULL, NULL, 0, NULL}
};

PY_INIT_MOD(cpp_impl, NULL, methods)