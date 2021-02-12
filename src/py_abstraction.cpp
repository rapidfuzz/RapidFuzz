/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#include "py_common.hpp"
#include "py_fuzz.hpp"
#include "py_string_metric.hpp"
#include "py_process.hpp"
#include "py_utils.hpp"


PyTypeObject PyExtractIter_Type {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "extract_iter",                   /* tp_name */
    sizeof(ExtractIterState),         /* tp_basicsize */
    0,                                /* tp_itemsize */
    (destructor)extract_iter_dealloc, /* tp_dealloc */
    0,                                /* tp_print */
    0,                                /* tp_getattr */
    0,                                /* tp_setattr */
    0,                                /* tp_reserved */
    0,                                /* tp_repr */
    0,                                /* tp_as_number */
    0,                                /* tp_as_sequence */
    0,                                /* tp_as_mapping */
    0,                                /* tp_hash */
    0,                                /* tp_call */
    0,                                /* tp_str */
    0,                                /* tp_getattro */
    0,                                /* tp_setattro */
    0,                                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    extract_iter_docstring,           /* tp_doc */
    0,                                /* tp_traverse */
    0,                                /* tp_clear */
    0,                                /* tp_richcompare */
    0,                                /* tp_weaklistoffset */
    PyObject_SelfIter,                /* tp_iter */
    (iternextfunc)extract_iter_next,  /* tp_iternext */
    0,                                /* tp_methods */
    0,                                /* tp_members */
    0,                                /* tp_getset */
    0,                                /* tp_base */
    0,                                /* tp_dict */
    0,                                /* tp_descr_get */
    0,                                /* tp_descr_set */
    0,                                /* tp_dictoffset */
    0,                                /* tp_init */
    PyType_GenericAlloc,              /* tp_alloc */
    extract_iter_new,                 /* tp_new */
};

static PyMethodDef methods[] = {
    /* utils */
    PY_METHOD(default_process),
    /* string_metric */
    PY_METHOD(levenshtein),
    PY_METHOD(normalized_levenshtein),
    PY_METHOD(hamming),
    PY_METHOD(normalized_hamming),
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
    PY_METHOD(extract),
    /* sentinel */
    {NULL, NULL, 0, NULL}};


#if PY_VERSION_HEX < PYTHON_VERSION(3, 0, 0)

PyMODINIT_FUNC initcpp_impl(void)
{
  if (PyType_Ready(&PyExtractIter_Type) < 0) {
    return;
  }

  PyObject* module = Py_InitModule3("cpp_impl", methods, NULL);

  if (!module) {
    return;
  }

  Py_INCREF((PyObject *)&PyExtractIter_Type);
  PyModule_AddObject(module, "extract_iter", (PyObject *)&PyExtractIter_Type);
}

#else

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpp_impl",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_cpp_impl(void)
{
  if (PyType_Ready(&PyExtractIter_Type) < 0) {
    return NULL;
  }

  PyObject* module = PyModule_Create(&moduledef);

  if (!module) {
    return NULL;
  }

  Py_INCREF((PyObject *)&PyExtractIter_Type);
  if (PyModule_AddObject(module, "extract_iter", (PyObject *)&PyExtractIter_Type) < 0) {
    Py_DECREF(module);
    Py_DECREF((PyObject *)&PyExtractIter_Type);
    return NULL;
  }

  return module;
}

#endif
