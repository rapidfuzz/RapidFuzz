/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

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

template <typename Sentence>
static inline python_string default_process_string(Sentence&& str)
{
  return rutils::default_process(std::forward<Sentence>(str));
}

// C++11 does not support generic lambdas
template <typename MatchingFunc>
struct GenericRatioVisitor {
  GenericRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return MatchingFunc::call(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

// C++11 does not support generic lambdas
template <typename MatchingFunc>
struct GenericProcessedRatioVisitor {
  GenericProcessedRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return MatchingFunc::call(rutils::default_process(s1), rutils::default_process(s2),
                              m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

template <typename MatchingFunc>
static inline PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
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

  if (non_default_process(processor)) {
    PyObject* proc_s1 = PyObject_CallFunctionObjArgs(processor, py_s1, NULL);
    if (proc_s1 == NULL) {
      return NULL;
    }

    PyObject* proc_s2 = PyObject_CallFunctionObjArgs(processor, py_s2, NULL);
    if (proc_s2 == NULL) {
      Py_DecRef(proc_s1);
      return NULL;
    }

    if (!valid_str(proc_s1, "s1") || !valid_str(proc_s2, "s2")) {
      return NULL;
    }

    auto s1_view = decode_python_string_view(proc_s1);
    auto s2_view = decode_python_string_view(proc_s2);

    double result = mpark::visit(GenericRatioVisitor<MatchingFunc>(score_cutoff), s1_view, s2_view);

    Py_DecRef(proc_s1);
    Py_DecRef(proc_s2);

    return PyFloat_FromDouble(result);
  }

  if (!valid_str(py_s1, "s1") || !valid_str(py_s2, "s2")) {
    return NULL;
  }

  auto s1_view = decode_python_string_view(py_s1);
  auto s2_view = decode_python_string_view(py_s2);

  double result;
  if (use_preprocessing(processor, processor_default)) {
    result =
        mpark::visit(GenericProcessedRatioVisitor<MatchingFunc>(score_cutoff), s1_view, s2_view);
  }
  else {
    result = mpark::visit(GenericRatioVisitor<MatchingFunc>(score_cutoff), s1_view, s2_view);
  }

  return PyFloat_FromDouble(result);
}

#define FUZZ_FUNC(name, process_default, docstring)                                                \
  PyDoc_STRVAR(name##_docstring, docstring);                                                       \
                                                                                                   \
  struct name##_func {                                                                             \
    template <typename... Args>                                                                    \
    static double call(Args&&... args)                                                             \
    {                                                                                              \
      return rfuzz::name(std::forward<Args>(args)...);                                             \
    }                                                                                              \
  };                                                                                               \
                                                                                                   \
  static PyObject* name(PyObject* /*self*/, PyObject* args, PyObject* keywds)                      \
  {                                                                                                \
    return fuzz_call<name##_func>(process_default, args, keywds);                                  \
  }

struct CachedFuzz {
  /* deleting polymorphic object without virtual destructur results in undefined behaviour */
  virtual ~CachedFuzz() = default;

  virtual void str1_set(python_string str)
  {
    m_str1 = std::move(str);
  }

  virtual void str2_set(python_string str)
  {
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
    "    96.55171966552734")

// C++11 does not support generic lambdas
struct RatioVisitor {
  RatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(RatioVisitor(score_cutoff), m_str1, m_str2);
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
    "    100")

// C++11 does not support generic lambdas
struct PartialRatioVisitor {
  PartialRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::partial_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedPartialRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(PartialRatioVisitor(score_cutoff), m_str1, m_str2);
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
    "    100.0")

// C++11 does not support generic lambdas
struct SortedSplitVisitor {
  template <typename Sentence>
  python_string operator()(Sentence&& s) const
  {
    return rutils::sorted_split(s).join();
  }
};

struct CachedTokenSortRatio : public CachedFuzz {
  void str1_set(python_string str) override
  {
    m_str1 = mpark::visit(SortedSplitVisitor(), str);
  }

  virtual void str2_set(python_string str) override
  {
    m_str2 = mpark::visit(SortedSplitVisitor(), str);
  }

  double call(double score_cutoff) override
  {
    return mpark::visit(RatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(partial_token_sort_ratio, true,
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
          "    float: ratio between s1 and s2 as a float between 0 and 100")

struct CachedPartialTokenSortRatio : public CachedFuzz {
  void str1_set(python_string str) override
  {
    m_str1 = mpark::visit(SortedSplitVisitor(), str);
  }

  virtual void str2_set(python_string str) override
  {
    m_str2 = mpark::visit(SortedSplitVisitor(), str);
  }

  double call(double score_cutoff) override
  {
    return mpark::visit(PartialRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(token_set_ratio, true,
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
          "    100.0")

// C++11 does not support generic lambdas
struct TokenSetRatioVisitor {
  TokenSetRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::token_set_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedTokenSetRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(TokenSetRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(partial_token_set_ratio, true,
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
          "    float: ratio between s1 and s2 as a float between 0 and 100")

// C++11 does not support generic lambdas
struct PartialTokenSetRatioVisitor {
  PartialTokenSetRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::partial_token_set_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedPartialTokenSetRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(PartialTokenSetRatioVisitor(score_cutoff), m_str1, m_str2);
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
    "    float: ratio between s1 and s2 as a float between 0 and 100")

// C++11 does not support generic lambdas
struct TokenRatioVisitor {
  TokenRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::token_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedTokenRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(TokenRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(partial_token_ratio, true,
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
          "    float: ratio between s1 and s2 as a float between 0 and 100")

// C++11 does not support generic lambdas
struct PartialTokenRatioVisitor {
  PartialTokenRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::partial_token_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedPartialTokenRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(PartialTokenRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(WRatio, true,
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
          "    float: ratio between s1 and s2 as a float between 0 and 100")

// C++11 does not support generic lambdas
struct WRatioVisitor {
  WRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::WRatio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedWRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(WRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(QRatio, true,
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
          "    96.55171966552734")

// C++11 does not support generic lambdas
struct QRatioVisitor {
  QRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::QRatio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedQRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(QRatioVisitor(score_cutoff), m_str1, m_str2);
  }
};

FUZZ_FUNC(quick_lev_ratio, true,
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
          "    float: ratio between s1 and s2 as a float between 0 and 100")

// C++11 does not support generic lambdas
struct QuickLevRatioVisitor {
  QuickLevRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const
  {
    return rfuzz::quick_lev_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

struct CachedQuickLevRatio : public CachedFuzz {
  double call(double score_cutoff) override
  {
    return mpark::visit(QuickLevRatioVisitor(score_cutoff), m_str1, m_str2);
  }
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

  /* this is pretty verbose. However it is faster than std::variant + std::visit */
#ifdef PYTHON_2
  if (PyObject_TypeCheck(py_sentence, &PyString_Type)) {
    Py_ssize_t len = PyString_GET_SIZE(py_sentence);
    char* str = PyString_AS_STRING(py_sentence);

    auto proc_str = rutils::default_process(rapidfuzz::basic_string_view<char>(str, len));
    return PyString_FromStringAndSize(proc_str.data(), proc_str.size());
  }
  else {
    Py_ssize_t len = PyUnicode_GET_SIZE(py_sentence);
    const Py_UNICODE* str = PyUnicode_AS_UNICODE(py_sentence);

    auto proc_str = rutils::default_process(rapidfuzz::basic_string_view<Py_UNICODE>(str, len));
    return PyUnicode_FromUnicode(proc_str.data(), proc_str.size());
  }
#else /* Python 3 */

  Py_ssize_t len = PyUnicode_GET_LENGTH(py_sentence);
  void* str = PyUnicode_DATA(py_sentence);

  switch (PyUnicode_KIND(py_sentence)) {
  case PyUnicode_1BYTE_KIND:
  {
    auto proc_str = rutils::default_process(
        rapidfuzz::basic_string_view<uint8_t>(static_cast<uint8_t*>(str), len));
    return PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, proc_str.data(), proc_str.size());
  }
  case PyUnicode_2BYTE_KIND:
  {
    auto proc_str = rutils::default_process(
        rapidfuzz::basic_string_view<uint16_t>(static_cast<uint16_t*>(str), len));
    return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, proc_str.data(), proc_str.size());
  }
  default:
  {
    auto proc_str = rutils::default_process(
        rapidfuzz::basic_string_view<uint32_t>(static_cast<uint32_t*>(str), len));
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, proc_str.data(), proc_str.size());
  }
  }
#endif
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

std::unique_ptr<CachedFuzz> get_matching_instance(PyObject* scorer)
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

constexpr const char* extractOne_docstring =
    "extractOne($module, query, choices, scorer = 'fuzz.WRatio', processor = "
    "'utils.default_process', score_cutoff = 0)\n"
    "--\n\n"
    "Find the best match in a list of choices\n\n"
    "Args:\n"
    "    query (str): string we want to find\n"
    "    choices (Iterable): list of all strings the query should be compared with or dict with a "
    "mapping\n"
    "        {<result>: <string to compare>}\n"
    "    scorer (Callable): optional callable that is used to calculate the matching score "
    "between\n"
    "        the query and each choice. WRatio is used by default\n"
    "    processor (Callable): optional callable that reformats the strings. "
    "utils.default_process\n"
    "        is used by default, which lowercases the strings and trims whitespace\n"
    "    score_cutoff (float): Optional argument for a score threshold. Matches with\n"
    "        a lower score than this number will not be returned. Defaults to 0\n\n"
    "Returns:\n"
    "    Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there "
    "is\n"
    "        no match with a score >= score_cutoff\n"
    "    Union[None, Tuple[str, float, Any]]: Returns the best match the best match\n"
    "        in form of a tuple or None when there is no match with a score >= score_cutoff. The "
    "Tuple will\n"
    "        be in the form`(<choice>, <ratio>, <index of choice>)` when `choices` is a list of "
    "strings\n"
    "        or `(<choice>, <ratio>, <key of choice>)` when `choices` is a mapping.";

static PyObject* extractOne(PyObject* /*self*/, PyObject* args, PyObject* keywds)
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
  PyObject* py_items;

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
    {NULL, NULL, 0, NULL}};

PY_INIT_MOD(cpp_impl, NULL, methods)
