# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_U64, RF_SCORER_FLAG_MULTI_STRING, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence, vector_slice
from libc.stdint cimport SIZE_MAX

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "edit_based.hpp":
    double normalized_indel_func( const RF_String&, const RF_String&, double) nogil except +

    size_t indel_func(const RF_String&, const RF_String&, size_t) nogil except +

    bool IndelInit(           RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False
    bool NormalizedIndelInit( RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False

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
    Calculates the minimum number of insertions and deletions
    required to change one sequence into the other. This is equivalent to the
    Levenshtein distance with a substitution weight of 2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
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
    The following implementation is used with a worst-case performance of ``O([N/64]M)``.

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
      parallel. The algorithm is described by [1]_ and is extended with support
      for UTF32 in this implementation. The time complexity of this
      algorithm is ``O(N)``.

    - If the length of the shorter string is ≥ 64 after removing the common affix
      a blockwise implementation of the Hyyrös' lcs algorithm is used, which calculates
      the Levenshtein distance in parallel (64 characters at a time).
      The algorithm is described by [1]_. The time complexity of this
      algorithm is ``O([N/64]M)``.

    The following image shows a benchmark of the Indel distance in RapidFuzz
    and python-Levenshtein. Similar to the normal Levenshtein distance
    python-Levenshtein uses a implementation with a time complexity of ``O(NM)``,
    while RapidFuzz has a time complexity of ``O([N/64]M)``.

    .. image:: img/indel_levenshtein.svg


    References
    ----------
    .. [4] Hyyrö, Heikki. "Bit-Parallel LCS-length Computation Revisited"
           Proc. 15th Australasian Workshop on Combinatorial Algorithms (AWOCA 2004).

    Examples
    --------
    Find the Indel distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Indel
    >>> Indel.distance("lewenstein", "levenshtein")
    3

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> Indel.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef size_t c_score_cutoff = <size_t>-1 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return indel_func(s1_proc.string, s2_proc.string, c_score_cutoff)


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
    Indel.distance : Indel distance

    Notes
    -----
    The normalization of the Indel distance is performed as
    `distance(s1, s2) / (len(s1) + len(s2))`.

    Examples
    --------
    Find the normalized Indel distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Indel
    >>> Indel.normalized_distance("lewenstein", "levenshtein")
    0.85714285714285

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Indel.normalized_distance("lewenstein", "levenshtein", score_cutoff=0.9)
    0.0

    When a different processor is used s1 and s2 do not have to be strings

    >>> Indel.normalized_distance(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return normalized_indel_func(s1_proc.string, s2_proc.string, c_score_cutoff)

cdef bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

cdef bool GetScorerFlagsIndel(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_U64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.u64 = 0
    dereference(scorer_flags).worst_score.u64 = SIZE_MAX
    return True

cdef bool GetScorerFlagsNormalizedIndel(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer IndelContext
IndelContext.version = SCORER_STRUCT_VERSION
IndelContext.kwargs_init = NoKwargsInit
IndelContext.get_scorer_flags = GetScorerFlagsIndel
IndelContext.scorer_func_init = IndelInit
distance._RF_Scorer = PyCapsule_New(&IndelContext, NULL, NULL)

cdef RF_Scorer NormalizedIndelContext
NormalizedIndelContext.version = SCORER_STRUCT_VERSION
NormalizedIndelContext.kwargs_init = NoKwargsInit
NormalizedIndelContext.get_scorer_flags = GetScorerFlagsNormalizedIndel
NormalizedIndelContext.scorer_func_init = NormalizedIndelInit
normalized_distance._RF_Scorer = PyCapsule_New(&NormalizedIndelContext, NULL, NULL)

