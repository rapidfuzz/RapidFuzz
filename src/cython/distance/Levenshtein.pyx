# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

"""
The Levenshtein (edit) distance is a string metric to measure the
difference between two strings/sequences s1 and s2.
It's defined as the minimum number of insertions, deletions or
substitutions required to transform s1 into s2.
"""

from _initialize import Editops
from _initialize cimport Editops, RfEditops
from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_SYMMETRIC
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
    double levenshtein_normalized_distance_func(  const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_distance_func(            const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +
    double levenshtein_normalized_similarity_func(const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    int64_t levenshtein_similarity_func(          const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +

    RfEditops levenshtein_editops_func(const RF_String&, const RF_String&) nogil except +

    bool LevenshteinDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool LevenshteinNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

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
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

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
    return levenshtein_distance_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates the levenshtein similarity in the range [max, 0] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``max - distance``, where max is the maximal possible
    Levenshtein distance given the lengths of the sequences s1/s2 and the weights.

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
        considered as a result. If the similarity is smaller than score_cutoff,
        0 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    similarity : int
        similarity between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return levenshtein_similarity_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def normalized_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein distance in the range [1, 0] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``distance / max``, where max is the maximal possible
    Levenshtein distance given the lengths of the sequences s1/s2 and the weights.

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
        For norm_dist > score_cutoff 1.0 is returned instead. Default is 1.0,
        which deactivates this behaviour.

    Returns
    -------
    norm_dist : float
        normalized distance between s1 and s2 as a float between 1.0 and 0.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = 1.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return levenshtein_normalized_distance_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def normalized_similarity(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein similarity in the range [0, 1] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``1 - normalized_distance``

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
        For norm_sim < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    norm_sim : float
        normalized similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    Examples
    --------
    Find the normalized Levenshtein similarity between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein")
    0.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein", score_cutoff=0.85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein", weights=(1,1,2))
    0.85714285714285

    When a different processor is used s1 and s2 do not have to be strings

    >>> Levenshtein.normalized_similarity(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
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

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return levenshtein_normalized_similarity_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


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
    described [8]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [8] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
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
    described [9]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [9] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
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

cdef bool GetScorerFlagsDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0
    dereference(scorer_flags).worst_score.f64 = 1.0
    return True

cdef bool GetScorerFlagsNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LevenshteinDistanceContext
LevenshteinDistanceContext.version = SCORER_STRUCT_VERSION
LevenshteinDistanceContext.kwargs_init = LevenshteinKwargsInit
LevenshteinDistanceContext.get_scorer_flags = GetScorerFlagsDistance
LevenshteinDistanceContext.scorer_func_init = LevenshteinDistanceInit
distance._RF_Scorer = PyCapsule_New(&LevenshteinDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinSimilarityContext
LevenshteinSimilarityContext.version = SCORER_STRUCT_VERSION
LevenshteinSimilarityContext.kwargs_init = LevenshteinKwargsInit
LevenshteinSimilarityContext.get_scorer_flags = GetScorerFlagsSimilarity
LevenshteinSimilarityContext.scorer_func_init = LevenshteinSimilarityInit
similarity._RF_Scorer = PyCapsule_New(&LevenshteinSimilarityContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedDistanceContext
LevenshteinNormalizedDistanceContext.version = SCORER_STRUCT_VERSION
LevenshteinNormalizedDistanceContext.kwargs_init = LevenshteinKwargsInit
LevenshteinNormalizedDistanceContext.get_scorer_flags = GetScorerFlagsNormalizedDistance
LevenshteinNormalizedDistanceContext.scorer_func_init = LevenshteinNormalizedDistanceInit
normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer LevenshteinNormalizedSimilarityContext
LevenshteinNormalizedSimilarityContext.version = SCORER_STRUCT_VERSION
LevenshteinNormalizedSimilarityContext.kwargs_init = LevenshteinKwargsInit
LevenshteinNormalizedSimilarityContext.get_scorer_flags = GetScorerFlagsNormalizedSimilarity
LevenshteinNormalizedSimilarityContext.scorer_func_init = LevenshteinNormalizedSimilarityInit
normalized_distance._RF_Scorer = PyCapsule_New(&LevenshteinNormalizedSimilarityContext, NULL, NULL)
