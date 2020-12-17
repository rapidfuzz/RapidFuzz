/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include <Python.h>

PyDoc_STRVAR(levenshtein_docstring,
R"(levenshtein($module, s1, s2, insert_cost = 1, delete_cost = 1, replace_cost = 1)
--

Calculates the minimum number of insertions, deletions, and substitutions
required to change one sequence into the other according to Levenshtein with custom
costs for insertion, deletion and substitution

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    insert_cost (int): cost for insertions
    delete_cost (int): cost for deletions
    replace_cost (int): cost for substitutions

Returns:
    int: weighted levenshtein distance between s1 and s2
)");
PyObject* levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(normalized_levenshtein_docstring,
R"(normalized_levenshtein($module, s1, s2, insert_cost = 1, delete_cost = 1, replace_cost = 1, processor = False, score_cutoff = 0)
--

Calculates a normalized levenshtein distance using custom
costs for insertion, deletion and substitution. So far only the following
combinations are supported:
- insert_cost= 1, delete_cost = 1, replace_cost = 1
- insert_cost= 1, delete_cost = 1, replace_cost = 2

further combinations will be supported in the future

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    insert_cost (int): cost for insertions
    delete_cost (int): cost for deletions
    replace_cost (int): cost for substitutions
    processor (Union[bool, Callable]): optional callable that reformats the strings.
        None is used by default.
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Defaults to 0.

Returns:
    float: normalized weighted levenshtein distance between s1 and s2 as a float between 0 and 100
)");
PyObject* normalized_levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(hamming_docstring,
R"(hamming($module, s1, s2)
--

Calculates the Hamming distance between two strings.

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare

Returns:
    int: Hamming distance between s1 and s2
)");
PyObject* hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds);

PyDoc_STRVAR(normalized_hamming_docstring,
R"(normalized_hamming($module, s1, s2, processor = False, score_cutoff = 0)
--

Calculates a normalized hamming distance

Args:
    s1 (str):  first string to compare
    s2 (str):  second string to compare
    processor (Union[bool, Callable]): optional callable that reformats the strings.
        None is used by default.
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Defaults to 0.

Returns:
    float: normalized hamming distance between s1 and s2 as a float between 0 and 100
)");
PyObject* normalized_hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds);

struct CachedNormalizedHamming : public CachedScorer {
  double call(double score_cutoff) override;
};
