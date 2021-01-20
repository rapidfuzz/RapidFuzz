/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "fuzz.hpp"
#include "py_utils.hpp"
#include "py_fuzz.hpp"
#include "utils.hpp"
#include <string>

namespace fuzz = rapidfuzz::fuzz;

// C++11 does not support generic lambdas
template <typename MatchingFunc>
struct GenericRatioVisitor {
  GenericRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return MatchingFunc::call(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

template <typename MatchingFunc>
static inline PyObject* fuzz_call(bool processor_default, PyObject* args, PyObject* keywds)
{
  PyObject* py_s1;
  PyObject* py_s2;
  PyObject* py_processor = NULL;
  double score_cutoff = 0;
  static const char* kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|Od", const_cast<char**>(kwlist), &py_s1,
                                   &py_s2, &py_processor, &score_cutoff))
  {
    return NULL;
  }

  if (py_s1 == Py_None || py_s2 == Py_None) {
    return PyFloat_FromDouble(0);
  }

  auto processor = get_processor(py_processor, processor_default);

  try {
    auto s1 = preprocess(py_s1, py_processor, processor, "s1");
    auto s2 = preprocess(py_s2, py_processor, processor, "s2");
    double result = mpark::visit(GenericRatioVisitor<MatchingFunc>(score_cutoff), s1.value, s2.value);
    return PyFloat_FromDouble(result);
  } catch(std::invalid_argument& e) {
    const char* msg = e.what();
    if (msg[0]) {
      PyErr_SetString(PyExc_ValueError, msg);
    }
    return NULL;
  }
}


/**********************************************
 *                   ratio
 *********************************************/

struct ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::ratio(std::forward<Args>(args)...);
  }
};

PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<ratio_func>(false, args, keywds);
}


/**********************************************
 *             partial_ratio
 *********************************************/

struct partial_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::partial_ratio(std::forward<Args>(args)...);
  }
};

PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
  return fuzz_call<partial_ratio_func>(false, args, keywds);
}


/**********************************************
 *             token_sort_ratio
 *********************************************/

struct token_sort_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::token_sort_ratio(std::forward<Args>(args)...);
  }
};

PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
  return fuzz_call<token_sort_ratio_func>(true, args, keywds);
}


/**********************************************
 *          partial_token_sort_ratio
 *********************************************/

struct partial_token_sort_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::partial_token_sort_ratio(std::forward<Args>(args)...);
  }
};

PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_sort_ratio_func>(true, args, keywds);
}


/**********************************************
 *               token_set_ratio
 *********************************************/

struct token_set_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::token_set_ratio(std::forward<Args>(args)...);
  }
};

PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<token_set_ratio_func>(true, args, keywds);
}


/**********************************************
 *          partial_token_set_ratio
 *********************************************/

struct partial_token_set_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::partial_token_set_ratio(std::forward<Args>(args)...);
  }
};

PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_set_ratio_func>(true, args, keywds);
}


/**********************************************
 *                token_ratio
 *********************************************/

struct token_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::token_ratio(std::forward<Args>(args)...);
  }
};

PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<token_ratio_func>(true, args, keywds);
}


/**********************************************
 *             partial_token_ratio
 *********************************************/

struct partial_token_ratio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::partial_token_ratio(std::forward<Args>(args)...);
  }
};

PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_ratio_func>(true, args, keywds);
}


/**********************************************
 *                    WRatio
 *********************************************/

struct WRatio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::WRatio(std::forward<Args>(args)...);
  }
};

PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<WRatio_func>(true, args, keywds);
}


/**********************************************
 *                    QRatio
 *********************************************/

struct QRatio_func {
  template <typename... Args>
  static double call(Args&&... args) {
    return fuzz::QRatio(std::forward<Args>(args)...);
  }
};

PyObject* QRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<QRatio_func>(true, args, keywds);
}
