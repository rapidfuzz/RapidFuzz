#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>
#include <string>
#include "fuzz.hpp"
#include "utils.hpp"


PyObject* ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* partial_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::partial_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::partial_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* token_sort_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::token_sort_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::token_sort_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* partial_token_sort_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::partial_token_sort_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::partial_token_sort_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* token_set_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::token_set_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::token_set_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}


PyObject* partial_token_set_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::partial_token_set_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::partial_token_set_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* token_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::token_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::token_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* partial_token_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::partial_token_ratio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::partial_token_ratio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}

PyObject* WRatio(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *s1;
    const wchar_t *s2;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"s1", "s2", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uu|fp", const_cast<char **>(kwlist),
                                     &s1, &s2, &score_cutoff, &preprocess))
        return NULL;

    double result;
    if (preprocess) {
        result = fuzz::WRatio(
            utils::default_process(s1),
            utils::default_process(s2),
            score_cutoff);
    } else {
        result = fuzz::WRatio(
            std::wstring_view(s1, wcslen(s1)),
            std::wstring_view(s2, wcslen(s2)),
            score_cutoff);
    }

    return PyFloat_FromDouble(result);
}


static PyMethodDef methods[] = {
    /* The cast of the function is necessary since PyCFunction values
     * only take two PyObject* parameters, and these functions take
     * three.
     */
    {"ratio", (PyCFunction)(void(*)(void))ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"partial_ratio", (PyCFunction)(void(*)(void))partial_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"token_sort_ratio", (PyCFunction)(void(*)(void))token_sort_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"partial_token_sort_ratio", (PyCFunction)(void(*)(void))partial_token_sort_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"token_set_ratio", (PyCFunction)(void(*)(void))token_set_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"partial_token_sort_ratio", (PyCFunction)(void(*)(void))partial_token_sort_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"token_ratio", (PyCFunction)(void(*)(void))token_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"partial_token_ratio", (PyCFunction)(void(*)(void))partial_token_ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"QRatio", (PyCFunction)(void(*)(void))ratio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
    {"WRatio", (PyCFunction)(void(*)(void))WRatio, METH_VARARGS | METH_KEYWORDS,
     R"pbdoc(
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
    )pbdoc"},
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