#pragma once

#include "fuzz.hpp"
#include "py_utils.hpp"
#include "utils.hpp"

namespace fuzz = rapidfuzz::fuzz;
namespace utils = rapidfuzz::utils;

PyDoc_STRVAR(fuzz_docstring,
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
);

struct ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {                                                 
    return rapidfuzz::fuzz::ratio(std::forward<Args>(args)...);
  }

  void preprocess_str1(python_string str) {
    m_str1 = str;
  }

  void preprocess_str2(python_string str) {
    m_str2 = str;
  }

  /* faster call after preprocess was called */
  double fast_call(score_cutoff) {
    return mpark::visit(
      [score_cutoff](auto&& val1, auto&& val2) {
          return rfuzz::ratio(val1, val2, score_cutoff);
        },
        m_str1, m_str2);
  }
private:
  python_string m_str1;
  python_string m_str2;
};

static PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds)
{
  return fuzz_call<fuzz_func>(false, args, keywds);
}