# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from _initialize import Editops
from _initialize cimport Editops, RfEditops
from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_MULTI_STRING, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence

from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdint cimport INT64_MAX, uint32_t, int64_t
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cdef struct LevenshteinWeightTable:
        int64_t insert_cost
        int64_t delete_cost
        int64_t replace_cost

cdef extern from "edit_based.hpp":
    double normalized_levenshtein_func( const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +

    int64_t levenshtein_func(const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +

    RfEditops levenshtein_editops_func(const RF_String&, const RF_String&) nogil except +

    bool LevenshteinInit(           RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool NormalizedLevenshteinInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

cdef inline void preprocess_strings(s1, s2, processor, RF_StringWrapper* s1_proc, RF_StringWrapper* s2_proc) except *:
    cdef RF_Preprocessor* preprocess_context = NULL

    if processor is None:
        s1_proc[0] = RF_StringWrapper(conv_sequence(s1))
        s2_proc[0] = RF_StringWrapper(conv_sequence(s2))
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            preprocess_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        if preprocess_context != NULL and preprocess_context.version == 1:
            preprocess_context.preprocess(s1, &(s1_proc[0].string))
            preprocess_context.preprocess(s2, &(s2_proc[0].string))
        else:
            s1 = processor(s1)
            s1_proc[0] = RF_StringWrapper(conv_sequence(s1), s1)
            s2 = processor(s2)
            s2_proc[0] = RF_StringWrapper(conv_sequence(s2), s2)

def distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates the minimum number of insertions, deletions, and substitutions
    required to change one sequence into the other according to Levenshtein with custom
    costs for insertion, deletion and substitution

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        max + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    Notes
    -----
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
        Hyyrös' lcs algorithm is used, which calculates the Indel distance in
        parallel. The algorithm is described by [4]_ and is extended with support
        for UTF32 in this implementation. The time complexity of this
        algorithm is ``O(N)``.

      - If the length of the shorter string is ≥ 64 after removing the common affix
        a blockwise implementation of the Hyyrös' lcs algorithm is used, which calculates
        the Levenshtein distance in parallel (64 characters at a time).
        The algorithm is described by [4]_. The time complexity of this
        algorithm is ``O([N/64]M)``.

    The following image shows a benchmark of the Indel distance in RapidFuzz
    and python-Levenshtein. Similar to the normal Levenshtein distance
    python-Levenshtein uses a implementation with a time complexity of ``O(NM)``,
    while RapidFuzz has a time complexity of ``O([N/64]M)``.

    .. image:: img/indel_levenshtein.svg


    Other weights:
      The implementation for other weights is based on Wagner-Fischer.
      It has a performance of ``O(N * M)`` and has a memory usage of ``O(N)``.
      Further details can be found in [2]_.

    References
    ----------
    .. [1] Hyyrö, Heikki. "A Bit-Vector Algorithm for Computing
           Levenshtein and Damerau Edit Distances."
           Nordic Journal of Computing, Volume 10 (2003): 29-39.
    .. [2] Wagner, Robert & Fischer, Michael
           "The String-to-String Correction Problem."
           J. ACM. 21. (1974): 168-173
    .. [3] Myers, Gene. "A fast bit-vector algorithm for approximate
           string matching based on dynamic programming."
           Journal of the ACM (JACM) 46.3 (1999): 395-415.
    .. [4] Hyyrö, Heikki. "Bit-Parallel LCS-length Computation Revisited"
           Proc. 15th Australasian Workshop on Combinatorial Algorithms (AWOCA 2004).

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> Levenshtein.distance("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.distance("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)

def editops(s1, s2, *, processor=None):
    """
    Return Editops describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    editops : Editops
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [1]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [1] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> for tag, src_pos, dest_pos in Levenshtein.editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[1] s2[0]
    replace s1[3] s2[2]
     insert s1[6] s2[5]
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    ops.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string)
    return ops

def opcodes(s1, s2, *, processor=None):
    """
    Return Opcodes describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    opcodes : Opcodes
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [1]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [1] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    
    >>> a = "qabxcd"
    >>> b = "abycdf"
    >>> for tag, i1, i2, j1, j2 in Levenshtein.opcodes("qabxcd", "abycdf"):
    ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
    ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
     delete a[0:1] (q) b[0:0] ()
      equal a[1:3] (ab) b[0:2] (ab)
    replace a[3:4] (x) b[2:3] (y)
      equal a[4:6] (cd) b[3:5] (cd)
     insert a[6:6] () b[5:6] (f)
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    ops.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string)
    return ops.as_opcodes()

def normalized_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein distance using custom
    costs for insertion, deletion and substitution.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    See Also
    --------
    Levenshtein.distance : Levenshtein distance

    Notes
    -----
    The normalization of the Levenshtein distance is performed in the following way:

    .. math::
      :nowrap:

      \\begin{align*}
        dist_{max} &= \\begin{cases}
          min(len(s1), len(s2)) \cdot sub,       & \\text{if } sub \leq ins + del \\\\
          len(s1) \cdot del + len(s2) \cdot ins, & \\text{otherwise}
        \end{cases}\\\\[10pt]

        dist_{max} &= \\begin{cases}
          dist_{max} + (len(s1) - len(s2)) \cdot del, & \\text{if } len(s1) > len(s2) \\\\
          dist_{max} + (len(s2) - len(s1)) \cdot ins, & \\text{if } len(s1) < len(s2) \\\\
          dist_{max},                                 & \\text{if } len(s1) = len(s2)
        \end{cases}\\\\[10pt]

        ratio &= \\frac{distance(s1, s2)}{dist_{max}}
      \end{align*}

    Examples
    --------
    Find the normalized Levenshtein distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein")
    0.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein", score_cutoff=0.85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein", weights=(1,1,2))
    0.85714285714285

     When a different processor is used s1 and s2 do not have to be strings

    >>> Levenshtein.normalized_distance(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return normalized_levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)

cdef void KwargsDeinit(RF_Kwargs* self):
    free(<void*>dereference(self).context)

cdef bool LevenshteinKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    cdef int64_t insertion, deletion, substitution
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>malloc(sizeof(LevenshteinWeightTable))

    if not weights:
        raise MemoryError

    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    dereference(weights).insert_cost = insertion
    dereference(weights).delete_cost = deletion
    dereference(weights).replace_cost = substitution
    dereference(self).context = weights
    dereference(self).dtor = KwargsDeinit
    return True

cdef bool GetScorerFlagsLevenshtein(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsNormalizedLevenshtein(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LevenshteinContext
LevenshteinContext.version = SCORER_STRUCT_VERSION
LevenshteinContext.kwargs_init = LevenshteinKwargsInit
LevenshteinContext.get_scorer_flags = GetScorerFlagsLevenshtein
LevenshteinContext.scorer_func_init = LevenshteinInit
distance._RF_Scorer = PyCapsule_New(&LevenshteinContext, NULL, NULL)

cdef RF_Scorer NormalizedLevenshteinContext
NormalizedLevenshteinContext.version = SCORER_STRUCT_VERSION
NormalizedLevenshteinContext.kwargs_init = LevenshteinKwargsInit
NormalizedLevenshteinContext.get_scorer_flags = GetScorerFlagsNormalizedLevenshtein
NormalizedLevenshteinContext.scorer_func_init = NormalizedLevenshteinInit
normalized_distance._RF_Scorer = PyCapsule_New(&NormalizedLevenshteinContext, NULL, NULL)

