#define PY_SSIZE_T_CLEAN  /* Make "s#" use Py_ssize_t rather than int. */
#include <Python.h>
#include <string>
#include <algorithm>
#include "process.hpp"
#include "utils.hpp"
#include "fuzz.hpp"


constexpr const char * extract_docstring = R"(
Find the best matches in a list of choices

Args: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    limit (int): maximum amount of results to return
    score_cutoff (float): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

Returns: 
    List[Tuple[str, float]]: returns a list of all matches that have a score >= score_cutoff
)";

PyObject* extract(PyObject *self, PyObject *args, PyObject *keywds) {
    const wchar_t *query;
    PyObject* py_choices;
	std::size_t limit = 5;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"query", "choices", "limit", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uO|nfp", const_cast<char **>(kwlist),
                                     &query, &py_choices, &limit, &score_cutoff, &preprocess)) {
        return NULL;
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
        return NULL;
    }

    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);
    std::wstring cleaned_query = (preprocess) ? utils::default_process(query) : std::wstring(query, wcslen(query));

    std::vector<std::pair<const wchar_t*, double> > results;
    results.reserve(choice_count);

    for (std::size_t i = 0; i < choice_count; ++i) {
        PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

        const wchar_t *choice;
        if (!PyArg_Parse(py_choice, "u", &choice)) {
            PyErr_SetString(PyExc_TypeError, "Choices must be a sequence of strings");
            Py_DECREF(choices);
            return NULL;
        }

        double score;
        if (preprocess) {
            score = fuzz::WRatio(
                cleaned_query,
                utils::default_process(choice),
                score_cutoff);
        } else {
            score = fuzz::WRatio(
                cleaned_query,
                std::wstring_view(choice, wcslen(choice)),
                score_cutoff);
        }

        if (score >= score_cutoff) {
            results.emplace_back(std::make_pair(choice, score));
        }
    }

    Py_DECREF(choices);

    std::sort(results.rbegin(), results.rend(), [](auto const& t1, auto const& t2) {
        return std::get<1>(t1) < std::get<1>(t2);
    });

    if (limit < results.size()) {
        results.resize(limit);
    }

    PyObject* py_return = PyList_New(results.size());

    for (std::size_t i = 0; i < results.size(); ++i) {
        auto const& [choice, score] = results[i];
        PyObject* py_tuple = Py_BuildValue("(ud)", choice, score);
        PyList_SetItem(py_return, i, py_tuple);
    }

    return py_return;
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
    const wchar_t *query;
    PyObject* py_choices;
    float score_cutoff = 0;
    bool preprocess = true;
    static const char *kwlist[] = {"query", "choices", "score_cutoff", "preprocess", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "uO|fp", const_cast<char **>(kwlist),
                                     &query, &py_choices, &score_cutoff, &preprocess)) {
        return NULL;
    }

    PyObject* choices = PySequence_Fast(py_choices, "Choices must be a sequence of strings");
    if (!choices) {
        return NULL;
    }

    std::size_t choice_count = PySequence_Fast_GET_SIZE(choices);
    std::wstring cleaned_query = (preprocess) ? utils::default_process(query) : std::wstring(query, wcslen(query));

    bool match_found = false;
    const wchar_t* result_choice;

    for (std::size_t i = 0; i < choice_count; ++i) {
        PyObject* py_choice = PySequence_Fast_GET_ITEM(choices, i);

        const wchar_t *choice;
        if (!PyArg_Parse(py_choice, "u", &choice)) {
            PyErr_SetString(PyExc_TypeError, "Choices must be a sequence of strings");
            Py_DECREF(choices);
            return NULL;
        }

        double score;
        if (preprocess) {
            score = fuzz::WRatio(
                cleaned_query,
                utils::default_process(choice),
                score_cutoff);
        } else {
            score = fuzz::WRatio(
                cleaned_query,
                std::wstring_view(choice, wcslen(choice)),
                score_cutoff);
        }

        if (score >= score_cutoff) {
            // increase the score_cutoff by a small step so it might be able to exit early
            score_cutoff = score + 0.00001;
            match_found = true;
            result_choice = choice;
        }
    }

    Py_DECREF(choices);

    if (!match_found) {
        Py_RETURN_NONE;
    }

    return Py_BuildValue("(ud)", result_choice, std::min(score_cutoff, static_cast<float>(100)));
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