#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>
#include <string>
#include <algorithm>
#include <boost/utility/string_view.hpp>
#include "utils.hpp"
#include "fuzz.hpp"
#include "py_utils.hpp"


constexpr const char * extract_docstring = R"(
Find the best matches in a list of choices

Args: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    score_cutoff (float): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

Returns: 
    List[Tuple[str, float]]: returns a list of all matches that have a score >= score_cutoff
)";

PyObject* extract(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_query;
    PyObject* py_choices;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"query", "choices", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UO|dp", const_cast<char **>(kwlist),
                                     &py_query, &py_choices, &score_cutoff, &preprocess)) {
        return NULL;
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
        return NULL;
    }
    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);

    if (PyUnicode_READY(py_query)) {
        return NULL;
    }

    std::wstring cleaned_query = PyObject_To_Wstring(py_query, preprocess);
    uint64_t query_bitmap = utils::bitmap_create(cleaned_query);

    PyObject* results = PyList_New(0);

    for (std::size_t i = 0; i < choice_count; ++i) {
        PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

        if (!PyUnicode_Check(py_choice)) {
            PyErr_SetString(PyExc_TypeError, "Choices must be a sequence of strings");
            Py_DECREF(choices);
            return NULL;
        }

        Py_ssize_t len = PyUnicode_GET_LENGTH(py_choice);
        wchar_t* buffer = PyUnicode_AsWideCharString(py_choice, &len);
        std::wstring choice(buffer, len);
        PyMem_Free(buffer);

        std::wstring cleaned_choice = (preprocess) ? utils::default_process(choice) : choice;
        uint64_t choice_bitmap = utils::bitmap_create(cleaned_choice);

        double score= fuzz::WRatio(
                Sentence<wchar_t>(cleaned_query, query_bitmap),
                Sentence<wchar_t>(cleaned_choice, choice_bitmap),
                score_cutoff);

        if (score >= score_cutoff) {
            PyList_Append(results, Py_BuildValue("(u#d)", choice.c_str(), choice.length(), score));
        }
    }

    Py_DECREF(choices);
    return results;
}


constexpr const char * extractOne_docstring = R"(
Find the best match in a list of choices

Args:
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    score_cutoff (float): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

Returns:
    Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there is
        no match with a score >= score_cutoff
)";

PyObject* extractOne(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *py_query;
    PyObject* py_choices;
    double score_cutoff = 0;
    int preprocess = 1;
    static const char *kwlist[] = {"query", "choices", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "UO|dp", const_cast<char **>(kwlist),
                                     &py_query, &py_choices, &score_cutoff, &preprocess)) {
        return NULL;
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
        return NULL;
    }
    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);

    if (PyUnicode_READY(py_query)) {
        return NULL;
    }

    std::wstring cleaned_query = PyObject_To_Wstring(py_query, preprocess);
    uint64_t query_bitmap = utils::bitmap_create(cleaned_query);

    double end_score = 0;
    std::wstring result_choice;

    for (std::size_t i = 0; i < choice_count; ++i) {
        PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

        if (!PyUnicode_Check(py_choice)) {
            PyErr_SetString(PyExc_TypeError, "Choices must be a sequence of strings");
            Py_DECREF(choices);
            return NULL;
        }

        Py_ssize_t len = PyUnicode_GET_LENGTH(py_choice);
        wchar_t* buffer = PyUnicode_AsWideCharString(py_choice, &len);
        std::wstring choice(buffer, len);
        PyMem_Free(buffer);

        std::wstring cleaned_choice = (preprocess) ? utils::default_process(choice) : choice;
        uint64_t choice_bitmap = utils::bitmap_create(cleaned_choice);

        double score = fuzz::WRatio(
                Sentence<wchar_t>(cleaned_query, query_bitmap),
                Sentence<wchar_t>(cleaned_choice, choice_bitmap),
                score_cutoff);

        if (score >= score_cutoff) {
            // increase the score_cutoff by a small step so it might be able to exit early
            score_cutoff = score + 0.00001;
            end_score = score;
            result_choice = std::move(choice);
        }
    }

    Py_DECREF(choices);

    if (!end_score) {
        Py_RETURN_NONE;
    }

    return Py_BuildValue("(u#d)", result_choice.c_str(), result_choice.length(), end_score);
}


/* The cast of the function is necessary since PyCFunction values
* only take two PyObject* parameters, and these functions take three.
*/
#define PY_METHOD(x) { #x, (PyCFunction)(void(*)(void))x, METH_VARARGS | METH_KEYWORDS, x##_docstring }
static PyMethodDef methods[] = {
    PY_METHOD(extract),
    PY_METHOD(extractOne),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rapidfuzz._process",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__process(void) {
    return PyModule_Create(&moduledef);
}