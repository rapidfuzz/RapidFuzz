/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"
#include "utils.hpp"

namespace utils = rapidfuzz::utils;


PyDoc_STRVAR(default_process_docstring,
R"(default_process($module, sentence)
--

This function preprocesses a string by:
- removing all non alphanumeric characters
- trimming whitespaces
- converting all characters to lower case

Right now this only affects characters lower than 256
(extended Ascii), while all other characters are not modified.
This should be enough for most western languages. Full Unicode
support will be added in a later release.

Parameters
----------
sentence : str
    String to preprocess

Returns
-------
processed_string : str
    processed string

)");
PyObject* default_process(PyObject* /*self*/, PyObject* args, PyObject* keywds);

struct DefaultProcessVisitor {
  template <typename Sentence>
  python_string operator()(Sentence&& s) const {
    return utils::default_process(std::forward<Sentence>(s));
  }
};

struct DefaultProcessor {
  static python_string call(PyObject* str, const char* name) {
    if (!valid_str(str, name)) throw std::invalid_argument("");
    return mpark::visit(DefaultProcessVisitor(), decode_python_string(str));
  }
};

using processor_func = mpark::variant<
  mpark::monostate,                                            /* No processor */
  PythonStringWrapper (*) (PyObject*, PyObject*, const char*), /* Python processor */
  python_string (*) (PyObject*, const char*)                   /* C++ processor */
>;

static inline processor_func get_processor(PyObject* processor, bool processor_default)
{
  if (processor == NULL) {
    if (processor_default) {
      return DefaultProcessor::call;
    }
    return mpark::monostate();
  }

  if (PyCFunction_Check(processor)) {
    if (PyCFunction_GetFunction(processor) == PY_FUNC_CAST(default_process)) {
      return DefaultProcessor::call;
    }
    // add new processors here
  }

  if (PyCallable_Check(processor)) {
    return PythonProcessor::call;
  }

  if (PyObject_IsTrue(processor)) {
    return DefaultProcessor::call;
  }

  return mpark::monostate();
}

static inline PythonStringWrapper preprocess(PyObject* py_str, PyObject* py_processor, processor_func processor, const char* name)
{
  switch(processor.index()) {
  case 0: /* No Processor */
  {
    if (!valid_str(py_str, name)) throw std::invalid_argument("");
    return PythonStringWrapper(decode_python_string(py_str), py_str);
  }
  case 1: /* Python processor */
  {
    return mpark::get<1>(processor)(py_processor, py_str, name);
  }
  case 2: /* C++ processor */
  {
    return PythonStringWrapper(mpark::get<2>(processor)(py_str, name));
  }
  }
}