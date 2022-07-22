# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64, RF_SCORER_FLAG_SYMMETRIC
)
# required for preprocess_strings
from rapidfuzz.utils import default_process
from array import array
from cpp_common cimport RfEditops, RfEditOp, EditType, RF_StringWrapper, preprocess_strings, NoKwargsInit, CreateScorerContext

from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.vector cimport vector
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

cdef extern from "string_metric_cpp.hpp":
    double normalized_levenshtein_func( const RF_String&, const RF_String&, int64_t, int64_t, int64_t, double) nogil except +
    double normalized_hamming_func(     const RF_String&, const RF_String&, double) nogil except +
    double jaro_similarity_func(        const RF_String&, const RF_String&, double) nogil except +
    double jaro_winkler_similarity_func(const RF_String&, const RF_String&, double, double) nogil except +

    int64_t levenshtein_func(const RF_String&, const RF_String&, int64_t, int64_t, int64_t, int64_t) nogil except +
    int64_t hamming_func(    const RF_String&, const RF_String&, int64_t) nogil except +

    RfEditops levenshtein_editops_func(const RF_String&, const RF_String&) nogil except +

    bool LevenshteinInit(           RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool NormalizedLevenshteinInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool HammingInit(               RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool NormalizedHammingInit(     RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool JaroSimilarityInit(        RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool JaroWinklerSimilarityInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def levenshtein(s1, s2, *, weights=(1,1,1), processor=None, max=None):
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
    max : int or None, optional
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

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.distance` instead.
        This function will be removed in v3.0.0.

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.string_metric import levenshtein
    >>> levenshtein("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> levenshtein("lewenstein", "levenshtein", max=1)
    2

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef int64_t c_max = INT64_MAX if max is None else max

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_max)

cdef str levenshtein_edit_type_to_str(EditType edit_type):
    if edit_type == EditType.Insert:
        return "insert"
    elif edit_type == EditType.Delete:
        return "delete"
    # possibly requires no-op in the future as well
    else:
        return "replace"

cdef list levenshtein_editops_to_list(const RfEditops& ops):
    cdef int64_t op_count = ops.size()
    cdef list result_list = PyList_New(<Py_ssize_t>op_count)
    for i in range(op_count):
        result_item = (levenshtein_edit_type_to_str(ops[i].type), ops[i].src_pos, ops[i].dest_pos)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list

def levenshtein_editops(s1, s2, *, processor=None):
    """
    Return list of 3-tuples describing how to turn s1 into s2.
    Each tuple is of the form (tag, src_pos, dest_pos).

    The tags are strings, with these meanings:
    'replace':  s1[src_pos] should be replaced by s2[dest_pos]
    'delete':   s1[src_pos] should be deleted.
    'insert':   s2[dest_pos] should be inserted at s1[src_pos].

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
    editops : list[]
        edit operations required to turn s1 into s2

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.editops` instead.
        This function will be removed in v3.0.0.

    Examples
    --------
    >>> from rapidfuzz.string_metric import levenshtein_editops
    >>> for tag, src_pos, dest_pos in levenshtein_editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[1] s2[0]
    replace s1[3] s2[2]
     insert s1[6] s2[5]
    """
    cdef RF_StringWrapper s1_proc, s2_proc

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return levenshtein_editops_to_list(
        levenshtein_editops_func(s1_proc.string, s2_proc.string)
    )

def normalized_levenshtein(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
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
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Levenshtein.normalized_similarity` instead.
        This function will be removed in v3.0.0.

    See Also
    --------
    levenshtein : Levenshtein distance

    Examples
    --------
    Find the normalized Levenshtein distance between two strings:

    >>> from rapidfuzz.string_metric import normalized_levenshtein
    >>> normalized_levenshtein("lewenstein", "levenshtein")
    81.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> normalized_levenshtein("lewenstein", "levenshtein", score_cutoff=85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> normalized_levenshtein("lewenstein", "levenshtein", weights=(1,1,2))
    85.71428571428571

     When a different processor is used s1 and s2 do not have to be strings

    >>> normalized_levenshtein(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    81.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef int64_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return normalized_levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)


def hamming(s1, s2, *, processor=None, max=None):
    """
    Calculates the Hamming distance between two strings.
    The hamming distance is defined as the number of positions
    where the two strings differ. It describes the minimum
    amount of substitutions required to transform s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    max : int or None, optional
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
        If s1 and s2 have a different length

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Hamming.distance` instead.
        This function will be removed in v3.0.0.
    """
    cdef int64_t c_max = INT64_MAX if max is None else max
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return hamming_func(s1_proc.string, s2_proc.string, c_max)


def normalized_hamming(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates a normalized hamming distance

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
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If s1 and s2 have a different length

    See Also
    --------
    hamming : Hamming distance

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Hamming.normalized_similarity` instead.
        This function will be removed in v3.0.0.
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return normalized_hamming_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def jaro_similarity(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the jaro similarity

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
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.Jaro.similarity` instead.
        This function will be removed in v3.0.0.
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return jaro_similarity_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def jaro_winkler_similarity(s1, s2, *, double prefix_weight=0.1, processor=None, score_cutoff=None):
    """
    Calculates the jaro winkler similarity

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    prefix_weight : float, optional
        Weight used for the common prefix of the two strings.
        Has to be between 0 and 0.25. Default is 0.1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Raises
    ------
    ValueError
        If prefix_weight is invalid

    .. deprecated:: 2.0.0
        Use :func:`rapidfuzz.distance.JaroWinkler.similarity` instead.
        This function will be removed in v3.0.0.
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return jaro_winkler_similarity_func(s1_proc.string, s2_proc.string, prefix_weight, c_score_cutoff)

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

cdef bool JaroWinklerKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    cdef double* prefix_weight = <double*>malloc(sizeof(double))

    if not prefix_weight:
        raise MemoryError

    prefix_weight[0] = kwargs.get("prefix_weight", 0.1)
    dereference(self).context = prefix_weight
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
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsHamming(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_I64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.i64 = 0
    dereference(scorer_flags).worst_score.i64 = INT64_MAX
    return True

cdef bool GetScorerFlagsNormalizedHamming(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsJaroSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsJaroWinklerSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True


cdef RF_Scorer LevenshteinContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsLevenshtein, LevenshteinInit)
levenshtein._RF_Scorer = PyCapsule_New(&LevenshteinContext, NULL, NULL)

cdef RF_Scorer NormalizedLevenshteinContext = CreateScorerContext(LevenshteinKwargsInit, GetScorerFlagsNormalizedLevenshtein, NormalizedLevenshteinInit)
normalized_levenshtein._RF_Scorer = PyCapsule_New(&NormalizedLevenshteinContext, NULL, NULL)

cdef RF_Scorer HammingContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsHamming, HammingInit)
hamming._RF_Scorer = PyCapsule_New(&HammingContext, NULL, NULL)

cdef RF_Scorer NormalizedHammingContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsNormalizedHamming, NormalizedHammingInit)
normalized_hamming._RF_Scorer = PyCapsule_New(&NormalizedHammingContext, NULL, NULL)

cdef RF_Scorer JaroSimilarityContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsJaroSimilarity, JaroSimilarityInit)
jaro_similarity._RF_Scorer = PyCapsule_New(&JaroSimilarityContext, NULL, NULL)

cdef RF_Scorer JaroWinklerSimilarityContext = CreateScorerContext(JaroWinklerKwargsInit, GetScorerFlagsJaroWinklerSimilarity, JaroWinklerSimilarityInit)
jaro_winkler_similarity._RF_Scorer = PyCapsule_New(&JaroWinklerSimilarityContext, NULL, NULL)

def _GetScorerFlagsDistance(**kwargs):
    return {"optimal_score": 0, "worst_score": 2**63 - 1}


def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 100, "worst_score": 0}


levenshtein._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsDistance}

normalized_levenshtein._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

hamming._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsDistance}

normalized_hamming._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

jaro_similarity._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}

jaro_winkler_similarity._RF_ScorerPy = {"get_scorer_flags": _GetScorerFlagsSimilarity}
