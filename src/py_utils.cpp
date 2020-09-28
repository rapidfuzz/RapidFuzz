/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "py_utils.hpp"
#include "utils.hpp"
#include <string>

namespace rutils = rapidfuzz::utils;

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

static PyMethodDef methods[] = {
    PY_METHOD(default_process), {NULL, NULL, 0, NULL} /* sentinel */
};

PY_INIT_MOD(utils, NULL, methods)