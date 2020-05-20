/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */
/* Copyright © 2011 Adam Cohen */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "fuzz.hpp"
#include "string_utils.hpp"
#include <boost/utility/string_view.hpp>
#include <boost/optional.hpp>

namespace fuzz = rapidfuzz::fuzz;
namespace string_utils = rapidfuzz::string_utils;

boost::optional<std::pair<wchar_t*, Py_ssize_t>> PyString_AsBuffer(PyObject* str, PyObject *processor) {
    PyObject *proc_str = PyObject_CallFunctionObjArgs(processor, str, NULL);
    if (proc_str == NULL) {
        return boost::none;
    }
    Py_ssize_t len = PyUnicode_GET_LENGTH(proc_str);
    wchar_t* buffer = PyUnicode_AsWideCharString(proc_str, &len);
    if (buffer == NULL) {
        Py_DecRef(proc_str);
        return boost::none;
    }
    Py_DecRef(proc_str);
    return std::make_pair(buffer, len);
}

boost::optional<std::pair<wchar_t*, Py_ssize_t>> PyString_AsBuffer(PyObject* str) {
    Py_ssize_t len = PyUnicode_GET_LENGTH(str);
    wchar_t* buffer = PyUnicode_AsWideCharString(str, &len);
    if (buffer == NULL) {
        return boost::none;
    }
    return std::make_pair(buffer, len);
}

bool use_preprocessing(PyObject* processor, bool processor_default) {
    return processor ? PyObject_IsTrue(processor) : processor_default;
}

template<typename T>
static PyObject* fuzz_impl(T&& scorer, bool processor_default, PyObject* args, PyObject* keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    PyObject *processor = NULL;
    double score_cutoff = 0;
    static const char *kwlist[] = {"s1", "s2", "processor", "score_cutoff", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|Od", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &processor, &score_cutoff)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    if (PyCallable_Check(processor)) {
        auto s1 = PyString_AsBuffer(py_s1, processor);
        if(!s1) {
            return NULL;
        }
        auto s2 = PyString_AsBuffer(py_s2, processor);
        if(!s2) {
            PyMem_Free(s1->first);
            return NULL;
        }
        auto result = scorer(
            boost::wstring_view(s1->first, s1->second),
            boost::wstring_view(s2->first, s2->second),
            score_cutoff);

        PyMem_Free(s1->first);
        PyMem_Free(s2->first);

        return PyFloat_FromDouble(result);

    }
    
    auto s1 = PyString_AsBuffer(py_s1);
    if(!s1) {
        return NULL;
    }
    auto s2 = PyString_AsBuffer(py_s2);
    if(!s2) {
        PyMem_Free(s1->first);
        return NULL;
    }

    double result;

    if (use_preprocessing(processor, processor_default)) {
        result = scorer(
            string_utils::default_process(std::wstring(s1->first, s1->second)),
            string_utils::default_process(std::wstring(s2->first, s2->second)),
            score_cutoff);
    } else {
        result = scorer(
            boost::wstring_view(s1->first, s1->second),
            boost::wstring_view(s2->first, s2->second),
            score_cutoff);
    }
    
    PyMem_Free(s1->first);
    PyMem_Free(s2->first);

    return PyFloat_FromDouble(result);
}


PyDoc_STRVAR(ratio_docstring,
"ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"calculates a simple ratio between two strings\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
"Example:\n"
"    >>> fuzz.ratio(\"this is a test\", \"this is a test!\")\n"
"    96.55171966552734"
);

static PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::ratio<boost::wstring_view, boost::wstring_view>, false, args, keywds);
}


PyDoc_STRVAR(partial_ratio_docstring,
"partial_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"calculates the fuzz.ratio of the optimal string alignment\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
"Example:\n"
"    >>> fuzz.partial_ratio(\"this is a test\", \"this is a test!\")\n"
"    100"
);

static PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::partial_ratio<boost::wstring_view, boost::wstring_view>, false, args, keywds);
}

PyDoc_STRVAR(token_sort_ratio_docstring,
"token_sort_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"sorts the words in the strings and calculates the fuzz.ratio between them\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
"Example:\n"
"    >>> fuzz.token_sort_ratio(\"fuzzy wuzzy was a bear\", \"wuzzy fuzzy was a bear\")\n"
"    100.0"
);

static PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::token_sort_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(partial_token_sort_ratio_docstring,
"partial_token_sort_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"sorts the words in the strings and calculates the fuzz.partial_ratio between them\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);

static PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::partial_token_sort_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(token_set_ratio_docstring,
"token_set_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Compares the words in the strings based on unique and common words between them using fuzz.ratio\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
"Example:\n"
"    >>> fuzz.token_sort_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")\n"
"    83.8709716796875\n"
"    >>> fuzz.token_set_ratio(\"fuzzy was a bear\", \"fuzzy fuzzy was a bear\")\n"
"    100.0"
);

static PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::token_set_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(partial_token_set_ratio_docstring,
"partial_token_set_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Compares the words in the strings based on unique and common words between them using fuzz.partial_ratio\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);


static PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::partial_token_set_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(token_ratio_docstring,
"token_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio\n"
"    (faster than manually executing the two functions)\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);

static PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::token_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(partial_token_ratio_docstring,
"partial_token_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Helper method that returns the maximum of fuzz.partial_token_set_ratio and fuzz.partial_token_sort_ratio\n"
"    (faster than manually executing the two functions)\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);

static PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::partial_token_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(WRatio_docstring,
"WRatio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Calculates a weighted ratio based on the other ratio algorithms\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);

static PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::WRatio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

PyDoc_STRVAR(QRatio_docstring,
"QRatio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"calculates a quick ratio between two strings using fuzz.ratio\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100\n\n"
"Example:\n"
"    >>> fuzz.QRatio(\"this is a test\", \"this is a test!\")\n"
"    96.55171966552734"
);

static PyObject* QRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::ratio<boost::wstring_view, boost::wstring_view>, false, args, keywds);
}

PyDoc_STRVAR(quick_lev_ratio_docstring,
"quick_lev_ratio($module, s1, s2, processor = False, score_cutoff = 0)\n"
"--\n\n"
"Calculates a quick estimation of fuzz.ratio by counting uncommon letters between the two sentences.\n"
"Guaranteed to be equal or higher than fuzz.ratio.\n"
"(internally used by fuzz.ratio when providing it with a score_cutoff to speed up the matching)\n\n"
"Args:\n"
"    s1 (str): first string to compare\n"
"    s2 (str): first string to compare\n"
"    processor (Union[bool, Callable]): optional callable that reformats the strings. utils.default_process\n"
"        is used by default, which lowercases the strings and trims whitespace\n"
"    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.\n"
"        For ratio < score_cutoff 0 is returned instead. Defaults to 0.\n\n"
"Returns:\n"
"    float: ratio between s1 and s2 as a float between 0 and 100"
);

static PyObject* quick_lev_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds) {
    return fuzz_impl(fuzz::quick_lev_ratio<boost::wstring_view, boost::wstring_view>, true, args, keywds);
}

/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/

#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, x##_docstring }
static PyMethodDef methods[] = {
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
    PY_METHOD(quick_lev_ratio),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz.fuzz",
    NULL,
    -1,
    methods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    NULL,     /* m_clear */
    NULL   /* m_free */
};

PyMODINIT_FUNC PyInit_fuzz(void) {
    return PyModule_Create(&moduledef);
}
