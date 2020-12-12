/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "py_string_metric.hpp"
#include "py_process.hpp"
#include "py_utils.hpp"

static PyMethodDef methods[] = {
    /* utils */
    PY_METHOD(default_process),
    /* string_metric */
    PY_METHOD(levenshtein),
    PY_METHOD(normalized_levenshtein),
    PY_METHOD(hamming),
    PY_METHOD(normalized_letter_frequency),
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
    /* process */
    PY_METHOD(extractOne),
    /* sentinel */
    {NULL, NULL, 0, NULL}};

PY_INIT_MOD(cpp_impl, NULL, methods)
