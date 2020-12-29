/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"

PyDoc_STRVAR(extractOne_docstring,
R"(extractOne($module, query, choices, scorer = 'fuzz.WRatio', processor = 'utils.default_process', score_cutoff = 0)
--

Find the best match in a list of choices

Args:
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer (Callable): optional callable that is used to calculate the matching score between
        the query and each choice. WRatio is used by default
    processor (Callable): optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff (float): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

Returns:
    Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there is
        no match with a score >= score_cutoff
    Union[None, Tuple[str, float, Any]]: Returns the best match the best match
        in form of a tuple or None when there is no match with a score >= score_cutoff.
        The Tuple will be in the form `(<choice>, <ratio>, <index of choice>)`
        when `choices` is a list of strings or `(<choice>, <ratio>, <key of choice>)`
        when `choices` is a mapping.
)");
PyObject* extractOne(PyObject* /*self*/, PyObject* args, PyObject* keywds);


typedef struct {
    PyObject_HEAD
    Py_ssize_t choice_index;
    Py_ssize_t choice_count;
    PyObject* choicesObj;
    PyObject* choices;
    bool is_dict;


    PythonStringWrapper query;
    PyObject* queryObj;

    std::unique_ptr<Processor> processor;
    PyObject* processorObj;
    std::unique_ptr<CachedScorer> scorer;
    PyObject* scorerObj;
    // used when scorer is a python function
    PyObject* argsObj;
    PyObject* kwargsObj;

    double score_cutoff;
} ExtractIterState;




PyObject* extract_iter_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
void extract_iter_dealloc(ExtractIterState *state);
PyObject* extract_iter_next(ExtractIterState *state);