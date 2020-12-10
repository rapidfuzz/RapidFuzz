/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"


PyDoc_STRVAR(default_process_docstring,
R"(default_process($module, sentence)
--


)");
PyObject* default_process(PyObject* /*self*/, PyObject* args, PyObject* keywds);
