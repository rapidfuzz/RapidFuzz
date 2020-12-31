/* SPDX-License-Identifier: MIT */
/* Copyright © 2020 Max Bachmann */

#pragma once
#include <Python.h>

PyDoc_STRVAR(levenshtein_docstring,
R"(levenshtein($module, s1, s2, weights = (1, 1, 1), max = None)
--

Calculates the minimum number of insertions, deletions, and substitutions
required to change one sequence into the other according to Levenshtein with custom
costs for insertion, deletion and substitution

Parameters
----------
s1 : str
    First string to compare
s2 : str
    Second string to compare
weights : Tuple[int, int, int] or None, optional
    The weights for the three operations in the form
    (insertion, deletion, substitution). Default is (1, 1, 1),
    which gives all three operations a weight of 1.
max : int or None, optional
    Maximum Levenshtein distance between s1 and s2, that is
    considered as a result. If the distance is bigger than max,
    -1 is returned instead. Default is None, which deactivates
    this behaviour.

Returns
-------
distance : int
    levenshtein distance between s1 and s2

Notes
-----
Depending on the input parameters different optimised implementation are used
to improve the performance. Worst-case performance is ``O(m * n)``.
In the following, these are visualized in the form of simple diagrams. Inside
the diagrams ``ls`` refers to the longer string and ``ss`` refers to the
shorter string.

Insertion = 1, Deletion = 1, Substitution = 1:
  +==================+    Yes    +----------------------------+
  ||    max = 0     ||---------->| direct comparision (O(N))  |
  +==================+           +----------------------------+
        | No
        V
  +---------------------+
  | remove common affix |
  +---------------------+
        |
        V
  +==================+    Yes    +--------------------------+
  ||    max ≤ 3     ||---------->| mbleven algorithm (O(N)) |
  +==================+           +--------------------------+
        | No
        V
  +==================+    Yes    +--------------------------+
  || extended Ascii ||---------->| Hyyrös' algorithm (O(N)) |
  || len(ss) ≤ 64   ||           | described by [1]_        |
  +==================+           +--------------------------+
        | No
        V
  +-------------------------------------------------------------+
  | Wagner-Fischer using Ukkonens optimisation                  |
  | described by [2]_                                           |
  | TODO: replace with Myers algorithm (with blocks)            |
  +-------------------------------------------------------------+

Insertion = 1, Deletion = 1, Substitution = 2:
  +==================+   Yes    +----------------------------+
  ||    max = 0     ||--------->| direct comparision (O(N))  |
  +==================+          +----------------------------+
        | No
        V
  +---------------------+
  | remove common affix |
  +---------------------+
        |
        V
  +==================+   Yes    +--------------------------+
  ||    max ≤ 4     ||--------->| mbleven algorithm (O(N)) |
  +==================+          +--------------------------+
        | No
        V
  +==================+    Yes    +-------------------------------+
  ||  len(ss) ≤ 64  ||---------->| BitPAl algorithm (O(N))       |
  ||                ||           | described by [4]_             |
  ||                ||           | with additional UTF32 support |
  +==================+           +-------------------------------+
        | No
        V
  +==================+    Yes    +----------------------------+
  || extended Ascii ||---------->| BitPAl algorithm blockwise |
  ||                ||           | (O(N*M/64))                |
  +==================+           +----------------------------+
        | No
        V
  +--------------------------------------------------------------+
  | Wagner-Fischer using Ukkonens optimisation (O(N*M))          |
  | described by [2]_                                            |
  | TODO: add unicode support to blockwise BitPAL as replacement |
  +--------------------------------------------------------------+


Other weights:
  The implementation for other weights is based on Wagner-Fischer.
  It has a performance of ``O(m * n)`` and has a memory usage of ``O(n)``.
  Further details can be found in [2]_.


References
----------
.. [1] Hyyrö, Heikki. "A Bit-Vector Algorithm for Computing
       Levenshtein and Damerau Edit Distances."
       Nordic Journal of Computing, Volume 10 (2003): 29-39.
.. [2] Wagner, Robert & Fischer, Michael
       "The String-to-String Correction Problem."
       J. ACM. 21. (1974): 168-173
.. [3] Ukkonen, Esko. "Algorithms for Approximate String Matching."
       Information and Control. 64. (1985): 100-118
.. [4] Loving, Joshua & Hernández, Yözen & Benson, Gary.
       "BitPAl: A Bit-Parallel, General Integer-Scoring Sequence
       Alignment Algorithm. Bioinformatics"
       Bioinformatics, Volume 30 (2014): 3166–3173
.. [5] Myers, Gene. "A fast bit-vector algorithm for approximate
       string matching based on dynamic programming."
       Journal of the ACM (JACM) 46.3 (1999): 395-415.


Examples
--------
Find the Levenshtein distance between two strings:
>>> from rapidfuzz.string_metric import levenshtein
>>> levenshtein("lewenstein", "levenshtein")
2
Setting a maximum distance allows the implementation to select
a more efficient implementation:
>>> levenshtein("lewenstein", "levenshtein", max=1)
-1
It is possible to select different weights by passing a `weight`
tuple. Internally s1 and s2 might be swapped, so insertion and deletion
cost should usually have the same value.
>>> levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
3
)");

PyObject* levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(normalized_levenshtein_docstring,
R"(normalized_levenshtein($module, s1, s2, weights = (1, 1, 1), processor = None, score_cutoff = 0)
--

Calculates a normalized levenshtein distance using custom
costs for insertion, deletion and substitution. So far only the following
combinations are supported:
- weights = (1, 1, 1)
- weights = (1, 1, 2)

further combinations might be supported in the future

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.
weights : Tuple[int, int, int] or None, optional
    The weights for the three operations in the form
    (insertion, deletion, substitution). Default is (1, 1, 1),
    which gives all three operations a weight of 1.
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
    Normalized weighted levenshtein distance between s1 and s2
    as a float between 0 and 100

Raises
------
ValueError
    If unsupported weights are provided a ValueError is thrown

See Also
--------
levenshtein : Levenshtein distance

Notes
-----
Depending on the provided weights the normalisation is performed in different
ways:

Insertion = 1, Deletion = 1, Substitution = 1:
  .. math:: ratio = 100 \cdot \frac{distance(s1, s2)}{max(len(s1), len(s2))}

Insertion = 1, Deletion = 1, Substitution = 2:
  .. math:: ratio = 100 \cdot \frac{distance(s1, s2)}{len(s1) + len(s2)}

Different weights are currently not supported, since the library has no algorithm
for normalization yet.

Examples
--------
Find the normalized Levenshtein distance between two strings:
>>> from rapidfuzz.string_metric import normalized_levenshtein
>>> normalized_levenshtein("lewenstein", "levenshtein")
81.81818181818181
Setting a score_cutoff allows the implementation to select
a more efficient implementation:
>>> levenshtein("lewenstein", "levenshtein", score_cutoff=85)
0.0
It is possible to select different weights by passing a `weight`
tuple. Internally s1 and s2 might be swapped, so insertion and deletion
cost should usually have the same value.
>>> levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
85.71428571428571
When a different processor is used s1 and s2 do not have to be strings
>> levenshtein(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
81.81818181818181

)");
PyObject* normalized_levenshtein(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(hamming_docstring,
R"(hamming($module, s1, s2)
--

Calculates the Hamming distance between two strings.

Parameters
----------
s1 : str
    First string to compare.
s2 : str
    Second string to compare.

Returns
-------
distance : int
    Hamming distance between s1 and s2

)");
PyObject* hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds);


PyDoc_STRVAR(normalized_hamming_docstring,
R"(normalized_hamming($module, s1, s2, processor = None, score_cutoff = 0)
--

Calculates a normalized hamming distance

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
    Normalized hamming distance between s1 and s2
    as a float between 0 and 100
)");
PyObject* normalized_hamming(PyObject* /*self*/, PyObject* args, PyObject* keywds);
