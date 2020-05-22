/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <nonstd/string_view.hpp>
#include <variant/variant.hpp>


using python_string_view  = mpark::variant<
  nonstd::basic_string_view<uint8_t>,
  nonstd::basic_string_view<uint16_t>,
  nonstd::basic_string_view<uint32_t>
>;

python_string_view decode_python_string(PyObject* py_str) {
    Py_ssize_t len = PyUnicode_GET_LENGTH(py_str);
    void* str = PyUnicode_DATA(py_str);

    int str_kind = PyUnicode_KIND(py_str);

    switch (str_kind) {
    case PyUnicode_1BYTE_KIND:
        return nonstd::basic_string_view<uint8_t>(static_cast<uint8_t*>(str), len);
    case PyUnicode_2BYTE_KIND:
        return nonstd::basic_string_view<uint16_t>(static_cast<uint16_t*>(str), len);
    default:
        return nonstd::basic_string_view<uint32_t>(static_cast<uint32_t*>(str), len);
    }
}