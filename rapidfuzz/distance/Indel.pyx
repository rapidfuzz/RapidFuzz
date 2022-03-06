# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

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

cdef extern from "edit_based.hpp":
    double indel_normalized_distance_func(  const RF_String&, const RF_String&, double) nogil except +
    int64_t indel_distance_func(            const RF_String&, const RF_String&, int64_t) nogil except +
    double indel_normalized_similarity_func(const RF_String&, const RF_String&, double) nogil except +
    int64_t indel_similarity_func(          const RF_String&, const RF_String&, int64_t) nogil except +

    RfEditops indel_editops_func(const RF_String&, const RF_String&) nogil except +

    bool IndelDistanceInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelNormalizedDistanceInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelSimilarityInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool IndelNormalizedSimilarityInit(RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False


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

def distance(s1, s2, *, processor=None, score_cutoff=None):
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
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Examples
    --------
    Find the Indel distance between two strings:

    >>> from rapidfuzz.distance import Indel
    >>> Indel.distance("lewenstein", "levenshtein")
    3

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> Indel.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = INT64_MAX if score_cutoff is None else score_cutoff
    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return indel_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)



def similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the Indel similarity in the range [max, 0].

    This is calculated as ``(len1 + len2) - distance``.

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
        considered as a result. If the similarity is smaller than score_cutoff,
        0 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    similarity : int
        similarity between s1 and s2
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return indel_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein similarity in the range [1, 0].

    This is calculated as ``distance / (len1 + len2)``.

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
        For norm_dist > score_cutoff 1.0 is returned instead. Default is 1.0,
        which deactivates this behaviour.

    Returns
    -------
    norm_dist : float
        normalized distance between s1 and s2 as a float between 0 and 1.0
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 1.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return indel_normalized_distance_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def normalized_similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized indel similarity in the range [0, 1].

    This is calculated as ``1 - normalized_distance``

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
        For norm_sim < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    norm_sim : float
        normalized similarity between s1 and s2 as a float between 0 and 1.0

    Examples
    --------
    Find the normalized Indel similarity between two strings:

    >>> from rapidfuzz.distance import Indel
    >>> Indel.normalized_similarity("lewenstein", "levenshtein")
    0.85714285714285

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Indel.normalized_similarity("lewenstein", "levenshtein", score_cutoff=0.9)
    0.0

    When a different processor is used s1 and s2 do not have to be strings

    >>> Indel.normalized_similarity(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    if c_score_cutoff < 0:
        raise ValueError("score_cutoff has to be >= 0")

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return indel_normalized_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)

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
    described [6]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [6] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import Indel
    >>> for tag, src_pos, dest_pos in Indel.editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[0] s2[0]
     delete s1[3] s2[2]
     insert s1[4] s2[2]
     insert s1[6] s2[5]
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    ops.editops = indel_editops_func(s1_proc.string, s2_proc.string)
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
    described [7]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [7] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import Indel

    >>> a = "qabxcd"
    >>> b = "abycdf"
    >>> for tag, i1, i2, j1, j2 in Indel.opcodes(a, b):
    ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
    ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
     delete a[0:1] (q) b[0:0] ()
      equal a[1:3] (ab) b[0:2] (ab)
     delete a[3:4] (x) b[2:2] ()
     insert a[4:4] () b[2:3] (y)
      equal a[4:6] (cd) b[3:5] (cd)
     insert a[6:6] () b[5:6] (f)
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef Editops ops = Editops.__new__(Editops)

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    ops.editops = indel_editops_func(s1_proc.string, s2_proc.string)
    return ops.as_opcodes()


cdef bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

cdef bool GetScorerFlagsIndelDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsIndelNormalizedDistance(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 0.0
    dereference(scorer_flags).worst_score.f64 = 1
    return True

cdef bool GetScorerFlagsIndelSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = INT64_MAX
    dereference(scorer_flags).worst_score.i64 = 0
    return True

cdef bool GetScorerFlagsIndelNormalizedSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer IndelDistanceContext
IndelDistanceContext.version = SCORER_STRUCT_VERSION
IndelDistanceContext.kwargs_init = NoKwargsInit
IndelDistanceContext.get_scorer_flags = GetScorerFlagsIndelDistance
IndelDistanceContext.scorer_func_init = IndelDistanceInit
distance._RF_Scorer = PyCapsule_New(&IndelDistanceContext, NULL, NULL)

cdef RF_Scorer IndelNormalizedDistanceContext
IndelNormalizedDistanceContext.version = SCORER_STRUCT_VERSION
IndelNormalizedDistanceContext.kwargs_init = NoKwargsInit
IndelNormalizedDistanceContext.get_scorer_flags = GetScorerFlagsIndelNormalizedDistance
IndelNormalizedDistanceContext.scorer_func_init = IndelNormalizedDistanceInit
normalized_distance._RF_Scorer = PyCapsule_New(&IndelNormalizedDistanceContext, NULL, NULL)

cdef RF_Scorer IndelSimilarityContext
IndelSimilarityContext.version = SCORER_STRUCT_VERSION
IndelSimilarityContext.kwargs_init = NoKwargsInit
IndelSimilarityContext.get_scorer_flags = GetScorerFlagsIndelSimilarity
IndelSimilarityContext.scorer_func_init = IndelSimilarityInit
similarity._RF_Scorer = PyCapsule_New(&IndelSimilarityContext, NULL, NULL)

cdef RF_Scorer IndelNormalizedSimilarityContext
IndelNormalizedSimilarityContext.version = SCORER_STRUCT_VERSION
IndelNormalizedSimilarityContext.kwargs_init = NoKwargsInit
IndelNormalizedSimilarityContext.get_scorer_flags = GetScorerFlagsIndelNormalizedSimilarity
IndelNormalizedSimilarityContext.scorer_func_init = IndelNormalizedSimilarityInit
normalized_similarity._RF_Scorer = PyCapsule_New(&IndelNormalizedSimilarityContext, NULL, NULL)
