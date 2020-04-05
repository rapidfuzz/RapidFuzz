#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>
#include <string>
#include "fuzz.hpp"
#include "utils.hpp"
#include "py_utils.hpp"


constexpr const char * ratio_docstring = R"(
calculates a simple ratio between two strings

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100

Example:
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
)";

static PyObject* ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);
    
    double result = fuzz::ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* partial_ratio_docstring = R"(
calculates a partial ratio between two strings

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100

Example:
    >>> fuzz.partial_ratio("this is a test", "this is a test!")
    100.0
)";

static PyObject* partial_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::partial_ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* token_sort_ratio_docstring = R"(
sorts the words in the string and calculates the fuzz.ratio between them

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100

Example:
    >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100.0
)";

static PyObject* token_sort_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::token_sort_ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* partial_token_sort_ratio_docstring = R"(
sorts the words in the strings and calculates the fuzz.partial_ratio between them

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)";

static PyObject* partial_token_sort_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::partial_token_sort_ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* token_set_ratio_docstring = R"(
Compares the words in the strings based on unique and common words between them using fuzz.ratio

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100

Example:
    >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    83.8709716796875
    >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100.0
)";

static PyObject* token_set_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::token_set_ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* partial_token_set_ratio_docstring = R"(
Compares the words in the strings based on unique and common words between them using fuzz.partial_ratio

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)";

static PyObject* partial_token_set_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    printf("%ls - %ls - %f\n", s1.c_str(), s2.c_str(), score_cutoff);
    double result = fuzz::partial_token_set_ratio(s1, s2, score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* token_ratio_docstring = R"(
Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
            (faster than manually executing the two functions)

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)";

static PyObject* token_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::token_ratio(
            Sentence<wchar_t>(s1),
            Sentence<wchar_t>(s2),
            score_cutoff);

    return PyFloat_FromDouble(result);
}


constexpr const char* partial_token_ratio_docstring = R"(
Helper method that returns the maximum of fuzz.partial_token_set_ratio and fuzz.partial_token_sort_ratio
    (faster than manually executing the two functions)

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)";

static PyObject* partial_token_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result;
    if (preprocess) {
        result = fuzz::partial_token_ratio(
            s1,
            s2,
            score_cutoff);
    } else {
        result = fuzz::partial_token_ratio(
            s1,
            s2,
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}


constexpr const char* QRatio_docstring = R"(
Calculates a weighted ratio based on the other ratio algorithms

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed 
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)";

static PyObject* QRatio(PyObject *self, PyObject *args, PyObject *keywds) {
    return ratio(self, args, keywds);
}


constexpr const char* WRatio_docstring = R"(
calculates a quick ratio between two strings using fuzz.ratio

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Defaults to 0.
    preprocess (bool): Optional argument to specify whether the strings should be preprocessed
        using utils.default_process. Defaults to True.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100

Example:
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
)";

static PyObject* WRatio(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_s1;
    PyObject *py_s2;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UU|dp", const_cast<char **>(kwlist),
                                     &py_s1, &py_s2, &score_cutoff, &preprocess)) {
        return NULL;
    }

    if (PyUnicode_READY(py_s1) || PyUnicode_READY(py_s2)) {
        return NULL;
    }

    std::wstring s1 = PyObject_To_Wstring(py_s1, preprocess);
    std::wstring s2 = PyObject_To_Wstring(py_s2, preprocess);

    double result = fuzz::WRatio(
            Sentence<wchar_t>(s1),
            Sentence<wchar_t>(s2),
            score_cutoff);

    return PyFloat_FromDouble(result);
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
    PY_METHOD(QRatio),
    PY_METHOD(WRatio),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz.fuzz",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_fuzz(void) {
    return PyModule_Create(&moduledef);
}