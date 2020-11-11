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

struct CachedFuzz {
  virtual void str1_set(python_string str) {
    m_str1 = std::move(str);
  }

  virtual void str2_set(python_string str) {
    m_str2 = std::move(str);
  }

  virtual double call(double score_cutoff) = 0;

protected:
  python_string m_str1;
  python_string m_str2;
};


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

struct CachedRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};


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

struct CachedPartialRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::partial_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedTokenSortRatio : public CachedFuzz {
  void str1_set(python_string str) override {
    m_str1 = mpark::visit(
      [](auto&& val) -> python_string {return rutils::sorted_split(val).join();}, str);
  }

  virtual void str2_set(python_string str) override {
    m_str2 = mpark::visit(
      [](auto&& val) -> python_string {return rutils::sorted_split(val).join();}, str);
  }

  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedPartialTokenSortRatio : public CachedFuzz {
  void str1_set(python_string str) override {
    m_str1 = mpark::visit(
      [](auto&& val) -> python_string {return rutils::sorted_split(val).join();}, str);
  }

  virtual void str2_set(python_string str) override {
    m_str2 = mpark::visit(
      [](auto&& val) -> python_string {return rutils::sorted_split(val).join();}, str);
  }

  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::partial__ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedTokenSetRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::token_set_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedPartialTokenSetRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::partial_token_set_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedTokenRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::token_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedPartialTokenRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::partial_token_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedWRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::WRatio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedQRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::QRatio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

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

struct CachedQuickLevRatio : public CachedFuzz {
  double call(double score_cutoff) override {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::quick_lev_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
};

struct CachedPyFunc : public CachedFuzz {
  CachedPyFunc(PyObject* scorer)
    : m_scorer(scorer) {}

  void str1_set(python_string str) override {
    m_str1 = std::move(str);
  }

  void str2_set(python_string str) override {
    m_str2 = std::move(str);
  }

  double call(double score_cutoff) override {
    m_scorer
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::token_ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
private:
  PyObject* m_scorer;
};

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


std::unique_ptr<CachedFuzz> get_matching_instance(PyObject* scorer)
{
  if (scorer) {
    if (PyCFunction_Check(scorer)) {
        auto scorer_func = PyCFunction_GetFunction(scorer);
        if (scorer_func == PY_FUNC_CAST(ratio))
        {
          return std::make_unique<CachedRatio>();
        } else if (scorer_func == PY_FUNC_CAST(partial_ratio)) {
          return std::make_unique<CachedPartialRatio>();
        } else if (scorer_func == PY_FUNC_CAST(token_sort_ratio)) {
          return std::make_unique<CachedTokenSortRatio>();
        } else if (scorer_func == PY_FUNC_CAST(token_set_ratio)) {
          return std::make_unique<CachedTokenSetRatio>();
        } else if (scorer_func == PY_FUNC_CAST(partial_token_sort_ratio)) {
          return std::make_unique<CachedPartialTokenSortRatio>();
        } else if (scorer_func == PY_FUNC_CAST(partial_token_set_ratio)) {
          return std::make_unique<CachedPartialTokenSetRatio>();
        } else if (scorer_func == PY_FUNC_CAST(token_ratio)) {
          return std::make_unique<CachedTokenRatio>();
        } else if (scorer_func == PY_FUNC_CAST(partial_token_ratio)) {
          return std::make_unique<CachedPartialTokenRatio>();
        } else if (scorer_func == PY_FUNC_CAST(WRatio)) {
          return std::make_unique<CachedWRatio>();
        } else if (scorer_func == PY_FUNC_CAST(QRatio)) {
          return std::make_unique<CachedQRatio>();
        }
    }
    /* call python function */
    return std::make_unique<CachedWRatio>();
  /* default is fuzz.WRatio */
  } else {
    return std::make_unique<CachedWRatio>();
  }
}



constexpr const char* extractOne_docstring = R"()";

static PyObject* extractOne(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  bool match_found = false;
  PyObject* result_choice = NULL;
  PyObject* choice_key = NULL;
  std::vector<PyObject*> outer_owner_list;
  python_string query;

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

  if (!valid_str(py_query, "query")) {
    return NULL;
  }

  if (!process_string(py_query, processor, true, query, outer_owner_list)) {
    return NULL;
  }

  auto scorer = get_matching_instance(py_scorer);
  scorer->str1_set(query);

  /* dict like container */
  if (PyObject_HasAttrString(py_choices, "items")) {
    //PyObject* py_items = PyObject_CallMethodObjArgs(py_choices, "items", NULL);//todo python3.9
    PyObject* py_items = PyObject_CallMethod(py_choices, "items", "");
    if (!py_items) {
      free_owner_list(outer_owner_list);
      return NULL;
    }
    outer_owner_list.push_back(py_items);

    PyObject* choices = PySequence_Fast(py_items, "Choices must be a sequence of strings");
    if (!choices) {
      free_owner_list(outer_owner_list);
      return NULL;
    }
    outer_owner_list.push_back(choices);

    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);

    for (std::size_t i = 0; i < choice_count; ++i) {

      PyObject* py_item = PySequence_Fast_GET_ITEM(choices, i);

      PyObject* py_choice = NULL;
      PyObject* py_match_choice = NULL;

      if (!PyArg_ParseTuple(py_item, "OO", &py_choice, &py_match_choice))
      {
        free_owner_list(outer_owner_list);
        return NULL;
      }

      if (py_match_choice == Py_None) {
        continue;
      }

      if (!valid_str(py_match_choice, "choice")) {
        free_owner_list(outer_owner_list);
        return NULL;
      }

      std::vector<PyObject*> inner_owner_list;
      python_string choice;

      if (!process_string(py_match_choice, processor, true, choice, inner_owner_list)) {
        free_owner_list(outer_owner_list);
        return NULL;
      }

      scorer->str2_set(choice);
      double score = scorer->call(score_cutoff);

      if (score >= score_cutoff) {
        // increase the value by a small step so it might be able to exit early
        score_cutoff = score + (float)0.00001;
        match_found = true;
        result_choice = py_match_choice;
        choice_key = py_choice;
      } 
      free_owner_list(inner_owner_list);
    }

    free_owner_list(outer_owner_list);
        
    if (!match_found) {
      Py_RETURN_NONE;
    }

    if (score_cutoff > 100) {
      score_cutoff = 100;
    }
    return Py_BuildValue("(OdO)", result_choice, score_cutoff, choice_key);
  }
  /* list like container */
  else {
    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
      free_owner_list(outer_owner_list);
      return NULL;
    }
    outer_owner_list.push_back(choices);

    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);

    for (std::size_t i = 0; i < choice_count; ++i) {
      PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

      if (py_choice == Py_None) {
        continue;
      }

      if (!valid_str(py_choice, "choice")) {
        free_owner_list(outer_owner_list);
        return NULL;
      }

      std::vector<PyObject*> inner_owner_list;
      python_string choice;

      if (!process_string(py_choice, processor, true, choice, inner_owner_list)) {
        free_owner_list(outer_owner_list);
        return NULL;
      }

      scorer->str2_set(choice);
      double score = scorer->call(score_cutoff);

      if (score >= score_cutoff) {
        // increase the value by a small step so it might be able to exit early
        score_cutoff = score + (float)0.00001;
        match_found = true;
        result_choice = py_choice;
      }
      free_owner_list(inner_owner_list);
    }

    free_owner_list(outer_owner_list);
        
    if (!match_found) {
      Py_RETURN_NONE;
    }

    if (score_cutoff > 100) {
      score_cutoff = 100;
    }
    return Py_BuildValue("(Od)", result_choice, score_cutoff);
  }
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
    PY_METHOD(extractOne),
/* sentinel */
    {NULL, NULL, 0, NULL}
};

PY_INIT_MOD(cpp_impl, NULL, methods)