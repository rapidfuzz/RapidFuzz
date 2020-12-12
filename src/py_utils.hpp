/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"
#include "utils.hpp"

namespace utils = rapidfuzz::utils;


PyDoc_STRVAR(default_process_docstring,
R"(default_process($module, sentence)
--


)");
PyObject* default_process(PyObject* /*self*/, PyObject* args, PyObject* keywds);

struct DefaultProcessVisitor {
  template <typename Sentence>
  python_string operator()(Sentence&& s) const {
    return utils::default_process(std::forward<Sentence>(s));
  }
};

struct DefaultProcessor : public Processor  {
  DefaultProcessor() {}
  PythonStringWrapper call(PyObject* str, const char* name) override {
    if (!valid_str(str, name)) throw std::invalid_argument("");
    return PythonStringWrapper(mpark::visit(DefaultProcessVisitor(), decode_python_string(str)));
  }
};


static inline std::unique_ptr<Processor> get_processor(PyObject* processor, bool processor_default)
{
  if (processor == NULL) {
    if (processor_default) {
      return std::unique_ptr<Processor>(new DefaultProcessor());
    }
    return std::unique_ptr<Processor>(new NoProcessor());
  }
  
  if (PyCFunction_Check(processor)) {
    if (PyCFunction_GetFunction(processor) == PY_FUNC_CAST(default_process)) {
      return std::unique_ptr<Processor>(new DefaultProcessor());
    }
    // add new processors here
  }

  if (PyCallable_Check(processor)) {
    return std::unique_ptr<Processor>(new PythonProcessor(processor));
  }

  if (PyObject_IsTrue(processor)) {
    return std::unique_ptr<Processor>(new DefaultProcessor());
  }

  return std::unique_ptr<Processor>(new NoProcessor());
}
