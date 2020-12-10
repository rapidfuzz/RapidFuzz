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
R"(normalized_levenshtein($module, s1, s2, insert_cost = 1, delete_cost = 1, replace_cost = 1, score_cutoff = 1)
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
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.

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


PyDoc_STRVAR(quick_lev_ratio_docstring,
R"(quick_lev_ratio($module, s1, s2, processor = False, score_cutoff = 0)
--

Calculates a quick estimation of fuzz.ratio by counting uncommon letters between the two sentences.
Guaranteed to be equal or higher than fuzz.ratio.
(internally used by fuzz.ratio when providing it with a score_cutoff to speed up the matching)

Args:
    s1 (str): first string to compare
    s2 (str): second string to compare
    processor (Union[bool, Callable]): optional callable that reformats the strings.
        utils.default_process is used by default, which lowercases the strings and trims whitespace
    score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Defaults to 0.

Returns:
    float: ratio between s1 and s2 as a float between 0 and 100
)");
//todo rename
PyObject* quick_lev_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);

struct CachedQuickLevRatio : public CachedScorer {
  double call(double score_cutoff) override;
};