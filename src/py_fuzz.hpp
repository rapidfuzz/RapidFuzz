/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include "py_common.hpp"


PyDoc_STRVAR(ratio_docstring,
R"(ratio($module, s1, s2, processor = None, score_cutoff = 0)
--

calculates a simple ratio between two strings. This is a simple wrapper
for string_metric.normalized_levenshtein using the weights:
- weights = (1, 1, 2)

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is None, which deactivates this behaviour.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100

Examples
--------
>>> fuzz.ratio("this is a test", "this is a test!")
96.55171966552734
)");
PyObject* ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);

PyDoc_STRVAR(partial_ratio_docstring,
R"(partial_ratio($module, s1, s2, processor = None, score_cutoff = 0)
--

calculates the fuzz.ratio of the optimal string alignment

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is None, which deactivates this behaviour.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100

Examples
--------
>>> fuzz.partial_ratio("this is a test", "this is a test!")
100.0
)");
PyObject* partial_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(token_sort_ratio_docstring,
R"(token_sort_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

sorts the words in the strings and calculates the fuzz.ratio between them

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100

Examples
--------
>>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
100.0
)");
PyObject* token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(partial_token_sort_ratio_docstring,
R"(partial_token_sort_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

sorts the words in the strings and calculates the fuzz.partial_ratio between them

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100
)");
PyObject* partial_token_sort_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(token_set_ratio_docstring,
R"(token_set_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

Compares the words in the strings based on unique and common words between them
using fuzz.ratio

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100

Examples
--------
>>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
83.8709716796875
>>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
100.0
)");
PyObject* token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(partial_token_set_ratio_docstring,
R"(partial_token_set_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

Compares the words in the strings based on unique and common words between them
using fuzz.partial_ratio

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100
)");
PyObject* partial_token_set_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(token_ratio_docstring,
R"(token_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
    (faster than manually executing the two functions)

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100
)");
PyObject* token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(partial_token_ratio_docstring,
R"(partial_token_ratio($module, s1, s2, processor = True, score_cutoff = 0)
--

Helper method that returns the maximum of fuzz.partial_token_set_ratio and
fuzz.partial_token_sort_ratio (faster than manually executing the two functions)

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100
)");
PyObject* partial_token_ratio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(WRatio_docstring,
R"(WRatio($module, s1, s2, processor = True, score_cutoff = 0)
--

Calculates a weighted ratio based on the other ratio algorithms

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100
)");
PyObject* WRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(QRatio_docstring,
R"(QRatio($module, s1, s2, processor = True, score_cutoff = 0)
--

Calculates a quick ratio between two strings using fuzz.ratio.
The only difference to fuzz.ratio is, that this preprocesses
the strings by default.

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
processor: bool or callable, optional
  Optional callable that is used to preprocess the strings before
  comparing them. When processor is True ``utils.default_process``
  is used. Default is True.
score_cutoff : float, optional
    Optional argument for a score threshold as a float between 0 and 100.
    For ratio < score_cutoff 0 is returned instead. Default is 0,
    which deactivates this behaviour.

Returns
-------
ratio : float
    ratio distance between s1 and s2 as a float between 0 and 100

Examples
--------
>>> fuzz.QRatio("this is a test", "this is a test!")
100.0
)");
PyObject* QRatio(PyObject* /*self*/, PyObject* args, PyObject* keywds);
