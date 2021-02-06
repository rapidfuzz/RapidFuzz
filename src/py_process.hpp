/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"
#include "py_utils.hpp"

PyDoc_STRVAR(extractOne_docstring,
R"(extractOne($module, query, choices, scorer = 'fuzz.WRatio', processor = 'utils.default_process', score_cutoff = 0)
--

Find the best match in a list of choices

Parameters
----------
query : str
    string we want to find
choices : Iterable
    list of all strings the query should be compared with or dict with a mapping
    {<result>: <string to compare>}
scorer : Callable, optional
    Optional callable that is used to calculate the matching score between
    the query and each choice. fuzz.WRatio is used by default
processor : Callable, optional
    Optional callable that reformats the strings.
    utils.default_process is used by default, which lowercases the strings and trims whitespace
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    Matches with a lower score than this number will be ignored. Default is 0,
    which deactivates this behaviour.

Returns
-------
Union[None, Tuple[str, float, Any]]
    Returns the best match the best match
    in form of a tuple or None when there is no match with a score >= score_cutoff.
    The Tuple will be in the form `(<choice>, <ratio>, <index of choice>)`
    when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
    when `choices` is a mapping.
)");
PyObject* extractOne(PyObject* /*self*/, PyObject* args, PyObject* keywds);

PyDoc_STRVAR(extract_docstring, "");
PyObject* extract(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(extract_iter_docstring,
R"(extract_iter($module, query, choices, scorer = 'fuzz.WRatio', processor = 'utils.default_process', score_cutoff = 0)
--

Find the best match in a list of choices

Parameters
----------
query : str
    string we want to find
choices : Iterable
    list of all strings the query should be compared with or dict with a mapping
    {<result>: <string to compare>}
scorer : Callable, optional
    Optional callable that is used to calculate the matching score between
    the query and each choice. fuzz.WRatio is used by default
processor : Callable, optional
    Optional callable that reformats the strings.
    utils.default_process is used by default, which lowercases the strings and trims whitespace
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    Matches with a lower score than this number will be ignored. Default is 0,
    which deactivates this behaviour.

Yields
-------
Tuple[str, float, Any]
    Yields similarity between the query and each choice in form of a tuple.
    The Tuple will be in the form `(<choice>, <ratio>, <index of choice>)`
    when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
    when `choices` is a mapping.
    Matches with a similarity, that is smaller than score_cutoff are skipped.
)");
typedef struct {
    PyObject_HEAD
    Py_ssize_t choice_index;
    Py_ssize_t choice_count;
    PyObject* choicesObj;
    PyObject* choices;
    bool is_dict;


    PythonStringWrapper query;
    PyObject* queryObj;

    processor_func processor;
    PyObject* processorObj;
    std::unique_ptr<CachedScorer> scorer;
    PyObject* scorerObj;
    // used when scorer is a python function
    PyObject* argsObj;
    PyObject* kwargsObj;

    double score_cutoff;
    PyObject* scoreCutoffObj;
} ExtractIterState;




PyObject* extract_iter_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
void extract_iter_dealloc(ExtractIterState *state);
PyObject* extract_iter_next(ExtractIterState *state);