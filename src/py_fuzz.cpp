/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "fuzz.hpp"
#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "utils.hpp"
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
    PyObject* proc_s1 = PyObject_CallFunctionObjArgs(processor, py_s2, NULL);
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

/**********************************************
 *                   ratio
 *********************************************/

struct ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<ratio_func>(false, args, keywds);
}

// C++11 does not support generic lambdas
struct RatioVisitor {
  RatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedRatio::call(double score_cutoff) {
  return mpark::visit(RatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *             partial_ratio
 *********************************************/

struct partial_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::partial_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_ratio_func>(false, args, keywds);
}

// C++11 does not support generic lambdas
struct PartialRatioVisitor {
  PartialRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::partial_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedPartialRatio::call(double score_cutoff) {
  return mpark::visit(PartialRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *             token_sort_ratio
 *********************************************/

struct token_sort_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::token_sort_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<token_sort_ratio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct SortedSplitVisitor {
  template <typename Sentence>
  python_string operator()(Sentence&& s) const {
    return rutils::sorted_split(s).join();
  }
};

void CachedTokenSortRatio::str1_set(python_string str) {
  m_str1 = mpark::visit(SortedSplitVisitor(), str);
}

void CachedTokenSortRatio::str2_set(python_string str) {
  m_str2 = mpark::visit(SortedSplitVisitor(), str);
}

double CachedTokenSortRatio::call(double score_cutoff) {
  return mpark::visit(RatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *          partial_token_sort_ratio
 *********************************************/

struct partial_token_sort_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::partial_token_sort_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_sort_ratio_func>(true, args, keywds);
}

void CachedPartialTokenSortRatio::str1_set(python_string str) {
  m_str1 = mpark::visit(SortedSplitVisitor(), str);
}

void CachedPartialTokenSortRatio::str2_set(python_string str) {
  m_str2 = mpark::visit(SortedSplitVisitor(), str);
}

double CachedPartialTokenSortRatio::call(double score_cutoff) {
  return mpark::visit(PartialRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *               token_set_ratio
 *********************************************/

struct token_set_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::token_set_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<token_set_ratio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct TokenSetRatioVisitor {
  TokenSetRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::token_set_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedTokenSetRatio::call(double score_cutoff) {
  return mpark::visit(TokenSetRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *          partial_token_set_ratio
 *********************************************/

struct partial_token_set_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::partial_token_set_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_set_ratio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct PartialTokenSetRatioVisitor {
  PartialTokenSetRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::partial_token_set_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedPartialTokenSetRatio::call(double score_cutoff) {
  return mpark::visit(PartialTokenSetRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *                token_ratio
 *********************************************/

struct token_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::token_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<token_ratio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct TokenRatioVisitor {
  TokenRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::token_ratio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedTokenRatio::call(double score_cutoff) {
  return mpark::visit(TokenRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *             partial_token_ratio
 *********************************************/

struct partial_token_ratio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::partial_token_ratio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<partial_token_ratio_func>(true, args, keywds);
}

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

double CachedPartialTokenRatio::call(double score_cutoff) {
  return mpark::visit(PartialTokenRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *                    WRatio
 *********************************************/

struct WRatio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::WRatio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<WRatio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct WRatioVisitor {
  WRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::WRatio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedWRatio::call(double score_cutoff) {
  return mpark::visit(WRatioVisitor(score_cutoff), m_str1, m_str2);
}


/**********************************************
 *                    QRatio
 *********************************************/

struct QRatio_func {
  template <typename... Args>                                              
  static double call(Args&&... args) {                                                                                
    return rfuzz::QRatio(std::forward<Args>(args)...);                              
  }                           
};       

PyObject* QRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {                                                                                                \
  return fuzz_call<QRatio_func>(true, args, keywds);
}

// C++11 does not support generic lambdas
struct QRatioVisitor {
  QRatioVisitor(double score_cutoff) : m_score_cutoff(score_cutoff)
  {}

  template <typename Sentence1, typename Sentence2>
  double operator()(Sentence1&& s1, Sentence2&& s2) const {
    return rfuzz::QRatio(s1, s2, m_score_cutoff);
  }

private:
  double m_score_cutoff;
};

double CachedQRatio::call(double score_cutoff) {
  return mpark::visit(QRatioVisitor(score_cutoff), m_str1, m_str2);
}
