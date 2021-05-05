# distutils: language=c++
# cython: language_level=3
# cython: binding=True

{% from "cpp_common.j2" import ratio_call, ratio_help, similarity_func, distance_func %}

from rapidfuzz.utils import default_process
from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence#, conv_sequence
from array import array
from libc.stdlib cimport malloc, free

cdef inline proc_string conv_sequence(seq):
    if is_valid_string(seq):
        return convert_string(seq)
    elif isinstance(seq, array):
        print("test")
        return hash_array(seq)
    else:
        return hash_sequence(seq)

cdef extern from "cpp_string_metric.hpp":
    object levenshtein_no_process(                proc_string, proc_string, size_t, size_t, size_t, size_t) nogil except +
    object levenshtein_default_process(           proc_string, proc_string, size_t, size_t, size_t, size_t) nogil except +

def levenshtein(s1, s2, weights=(1,1,1), processor=None, max=None):
{{ ratio_help(
ratio_type="distance",    
processor_default="None",
description=
"""
Calculates the minimum number of insertions, deletions, and substitutions
required to change one sequence into the other according to Levenshtein with custom
costs for insertion, deletion and substitution
""",
custom_params=
"""
weights : Tuple[int, int, int] or None, optional
    The weights for the three operations in the form
    (insertion, deletion, substitution). Default is (1, 1, 1),
    which gives all three operations a weight of 1.
""",
exceptions=
"""
ValueError
    If unsupported weights are provided a ValueError is thrown
""",
notes=
"""
Depending on the input parameters different optimized implementation are used
to improve the performance.

Insertion = Deletion = Substitution:
  This is known as uniform Levenshtein distance and is the distance most commonly
  referred to as Levenshtein distance. The following implementation is used
  with a worst-case performance of ``O([N/64]M)``.

  - if max is 0 the similarity can be calculated using a direct comparision,
    since no difference between the strings is allowed.  The time complexity of
    this algorithm is ``O(N)``.

  - A common prefix/suffix of the two compared strings does not affect
    the Levenshtein distance, so the affix is removed before calculating the
    similarity.

  - If max is ≤ 3 the mbleven algorithm is used. This algorithm
    checks all possible edit operations that are possible under
    the threshold `max`. The time complexity of this algorithm is ``O(N)``.

  - If the length of the shorter string is ≤ 64 after removing the common affix
    Hyyrös' algorithm is used, which calculates the Levenshtein distance in
    parallel. The algorithm is described by [1]_. The time complexity of this
    algorithm is ``O(N)``.

  - If the length of the shorter string is ≥ 64 after removing the common affix
    a blockwise implementation of Myers' algorithm is used, which calculates
    the Levenshtein distance in parallel (64 characters at a time).
    The algorithm is described by [3]_. The time complexity of this
    algorithm is ``O([N/64]M)``.

The following image shows a benchmark of the Levenshtein distance in multiple
Python libraries. All of them are implemented either in C/C++ or Cython.
The graph shows, that python-Levenshtein is the only library with a time
complexity of ``O(NM)``, while all other libraries have a time complexity of
``O([N/64]M)``. Especially for long strings RapidFuzz is a lot faster than
all the other tested libraries.

.. image:: img/uniform_levenshtein.svg


Insertion = Deletion, Substitution >= Insertion + Deletion:
  Since every Substitution can be performed as Insertion + Deletion, this variant
  of the Levenshtein distance only uses Insertions and Deletions. Therefore this
  variant is often referred to as InDel-Distance.  The following implementation
  is used with a worst-case performance of ``O([N/64]M)``.

  - if max is 0 the similarity can be calculated using a direct comparision,
    since no difference between the strings is allowed.  The time complexity of
    this algorithm is ``O(N)``.

  - if max is 1 and the two strings have a similar length, the similarity can be
    calculated using a direct comparision aswell, since a substitution would cause
    a edit distance higher than max. The time complexity of this algorithm
    is ``O(N)``.

  - A common prefix/suffix of the two compared strings does not affect
    the Levenshtein distance, so the affix is removed before calculating the
    similarity.

  - If max is ≤ 4 the mbleven algorithm is used. This algorithm
    checks all possible edit operations that are possible under
    the threshold `max`. As a difference to the normal Levenshtein distance this
    algorithm can even be used up to a threshold of 4 here, since the higher weight
    of substitutions decreases the amount of possible edit operations.
    The time complexity of this algorithm is ``O(N)``.

  - If the length of the shorter string is ≤ 64 after removing the common affix
    the BitPAl algorithm is used, which calculates the Levenshtein distance in
    parallel. The algorithm is described by [4]_ and is extended with support
    for UTF32 in this implementation. The time complexity of this
    algorithm is ``O(N)``.

  - If the length of the shorter string is ≥ 64 after removing the common affix
    a blockwise implementation of the BitPAl algorithm is used, which calculates
    the Levenshtein distance in parallel (64 characters at a time).
    The algorithm is described by [4]_. The time complexity of this
    algorithm is ``O([N/64]M)``.

The following image shows a benchmark of the InDel distance in RapidFuzz
and python-Levenshtein. Similar to the normal Levenshtein distance
python-Levenshtein uses a implementation with a time complexity of ``O(NM)``,
while RapidFuzz has a time complexity of ``O([N/64]M)``.

.. image:: img/indel_levenshtein.svg


Other weights:
  The implementation for other weights is based on Wagner-Fischer.
  It has a performance of ``O(N * M)`` and has a memory usage of ``O(N)``.
  Further details can be found in [2]_.
""",
references=
"""
.. [1] Hyyrö, Heikki. \"A Bit-Vector Algorithm for Computing
       Levenshtein and Damerau Edit Distances.\"
       Nordic Journal of Computing, Volume 10 (2003): 29-39.
.. [2] Wagner, Robert & Fischer, Michael
       \"The String-to-String Correction Problem.\"
       J. ACM. 21. (1974): 168-173
.. [3] Myers, Gene. \"A fast bit-vector algorithm for approximate
       string matching based on dynamic programming.\"
       Journal of the ACM (JACM) 46.3 (1999): 395-415.
.. [4] Loving, Joshua & Hernández, Yözen & Benson, Gary.
       \"BitPAl: A Bit-Parallel, General Integer-Scoring Sequence
       Alignment Algorithm. Bioinformatics\"
       Bioinformatics, Volume 30 (2014): 3166–3173
""",
examples=
"""
Find the Levenshtein distance between two strings:

>>> from rapidfuzz.string_metric import levenshtein
>>> levenshtein(\"lewenstein\", \"levenshtein\")
2

Setting a maximum distance allows the implementation to select
a more efficient implementation:

>>> levenshtein(\"lewenstein\", \"levenshtein\", max=1)
-1

It is possible to select different weights by passing a `weight`
tuple.

>>> levenshtein(\"lewenstein\", \"levenshtein\", weights=(1,1,2))
3
""") | indent(4, indentfirst=True) }}
    cdef size_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef size_t c_max = <size_t>-1 if max is None else max

    if processor is True or processor == default_process:
        {{ ratio_call("levenshtein_default_process", ["insertion", "deletion", "substitution", "c_max"]) | indent(8) }}
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    {{ ratio_call("levenshtein_no_process", ["insertion", "deletion", "substitution", "c_max"]) | indent(4) }}


cdef extern from "cpp_string_metric.hpp":
    double normalized_levenshtein_no_process(     proc_string, proc_string, size_t, size_t, size_t, double) nogil except +
    double normalized_levenshtein_default_process(proc_string, proc_string, size_t, size_t, size_t, double) nogil except +

def normalized_levenshtein(s1, s2, weights=(1,1,1), processor=None, score_cutoff=None):
{{ ratio_help(
ratio_type="similarity",    
processor_default="None",
description=
"""
Calculates a normalized levenshtein distance using custom
costs for insertion, deletion and substitution.
""",
custom_params=
"""
weights : Tuple[int, int, int] or None, optional
    The weights for the three operations in the form
    (insertion, deletion, substitution). Default is (1, 1, 1),
    which gives all three operations a weight of 1.
""",
exceptions=
"""
ValueError
    If unsupported weights are provided a ValueError is thrown
""",
see_also="levenshtein : Levenshtein distance",
notes=
"""
The normalization of the Levenshtein distance is performed in the following way:

.. math::
  :nowrap:

  \\\\begin{align*}
    dist_{max} &= \\\\begin{cases}
      min(len(s1), len(s2)) \\cdot sub,       & \\\\text{if } sub \\leq ins + del \\\\\\\\
      len(s1) \\cdot del + len(s2) \\cdot ins, & \\\\text{otherwise}
    \\end{cases}\\\\\\\\[10pt]

    dist_{max} &= \\\\begin{cases}
      dist_{max} + (len(s1) - len(s2)) \\cdot del, & \\\\text{if } len(s1) > len(s2) \\\\\\\\
      dist_{max} + (len(s2) - len(s1)) \\cdot ins, & \\\\text{if } len(s1) < len(s2) \\\\\\\\
      dist_{max},                                 & \\\\text{if } len(s1) = len(s2)
    \\end{cases}\\\\\\\\[10pt]

    ratio &= 100 \\cdot \\\\frac{distance(s1, s2)}{dist_{max}}
  \\end{align*}
""",
examples=
"""
Find the normalized Levenshtein distance between two strings:

>>> from rapidfuzz.string_metric import normalized_levenshtein
>>> normalized_levenshtein(\"lewenstein\", \"levenshtein\")
81.81818181818181

Setting a score_cutoff allows the implementation to select
a more efficient implementation:

>>> normalized_levenshtein(\"lewenstein\", \"levenshtein\", score_cutoff=85)
0.0

It is possible to select different weights by passing a `weight`
tuple.

>>> normalized_levenshtein(\"lewenstein\", \"levenshtein\", weights=(1,1,2))
85.71428571428571

 When a different processor is used s1 and s2 do not have to be strings

>>> normalized_levenshtein([\"lewenstein\"], [\"levenshtein\"], processor=lambda s: s[0])
81.81818181818181
""") | indent(4, indentfirst=True) }}
    if s1 is None or s2 is None:
        return 0

    cdef size_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if processor is True or processor == default_process:
        {{ ratio_call("normalized_levenshtein_default_process", ["insertion", "deletion", "substitution", "c_score_cutoff"]) | indent(8) }}
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    {{ ratio_call("normalized_levenshtein_no_process", ["insertion", "deletion", "substitution", "c_score_cutoff"]) | indent(4) }}


{{ distance_func(
ratio_name="hamming",
processor_default="None",
description=
"""
Calculates the Hamming distance between two strings.
The hamming distance is defined as the number of positions 
where the two strings differ. It describes the minimum
amount of substitutions required to transform s1 into s2.
""",
exceptions=
"""
ValueError
    If s1 and s2 have a different length
"""
)
}}

{{ similarity_func(
ratio_name="normalized_hamming",
processor_default="None",
description="Calculates a normalized hamming distance",
exceptions=
"""
ValueError
    If s1 and s2 have a different length
""",
see_also="hamming : Hamming distance"
)
}}
