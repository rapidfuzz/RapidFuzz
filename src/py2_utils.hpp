/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#define PY_SSIZE_T_CLEAN
#include "details/types.hpp"
#include <Python.h>
#include <variant/variant.hpp>

bool valid_str(PyObject* str, const char* name)
{
  if (!PyObject_TypeCheck(str, &PyString_Type) && !PyObject_TypeCheck(str, &PyUnicode_Type)) {
    PyErr_Format(PyExc_TypeError, "%s must be a String, Unicode or None", name);
    return false;
  }
  return true;
}

#define PY_INIT_MOD(name, doc, methods)                                                            \
  PyMODINIT_FUNC init##name(void)                                                                  \
  {                                                                                                \
    Py_InitModule3(#name, methods, doc);                                                           \
  }

using python_string =
    mpark::variant<std::basic_string<uint8_t>, std::basic_string<Py_UNICODE>,
                   rapidfuzz::basic_string_view<uint8_t>, rapidfuzz::basic_string_view<Py_UNICODE>>;

using python_string_view =
    mpark::variant<rapidfuzz::basic_string_view<uint8_t>, rapidfuzz::basic_string_view<Py_UNICODE>>;

python_string decode_python_string(PyObject* py_str)
{
  if (PyObject_TypeCheck(py_str, &PyString_Type)) {
    Py_ssize_t len = PyString_GET_SIZE(py_str);
    uint8_t* str = reinterpret_cast<uint8_t*>(PyString_AS_STRING(py_str));
    return rapidfuzz::basic_string_view<uint8_t>(str, len);
  }
  else {
    Py_ssize_t len = PyUnicode_GET_SIZE(py_str);
    Py_UNICODE* str = PyUnicode_AS_UNICODE(py_str);
    return rapidfuzz::basic_string_view<Py_UNICODE>(str, len);
  }
}

python_string_view decode_python_string_view(PyObject* py_str)
{
  if (PyObject_TypeCheck(py_str, &PyString_Type)) {
    Py_ssize_t len = PyString_GET_SIZE(py_str);
    uint8_t* str = reinterpret_cast<uint8_t*>(PyString_AS_STRING(py_str));
    return rapidfuzz::basic_string_view<uint8_t>(str, len);
  }
  else {
    Py_ssize_t len = PyUnicode_GET_SIZE(py_str);
    Py_UNICODE* str = PyUnicode_AS_UNICODE(py_str);
    return rapidfuzz::basic_string_view<Py_UNICODE>(str, len);
  }
}

PyObject* encode_python_string(rapidfuzz::basic_string_view<uint8_t> str)
{
  return PyString_FromStringAndSize(reinterpret_cast<const char*>(str.data()), str.size());
}

PyObject* encode_python_string(rapidfuzz::basic_string_view<Py_UNICODE> str)
{
  return PyUnicode_FromUnicode(str.data(), str.size());
}