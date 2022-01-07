# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_U64, RF_SCORER_FLAG_MULTI_STRING, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence
from libc.stdint cimport SIZE_MAX

from libcpp cimport bool
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "edit_based.hpp":
    double normalized_hamming_func(     const RF_String&, const RF_String&, double) nogil except +
    size_t hamming_func(    const RF_String&, const RF_String&, size_t) nogil except +

    bool HammingInit(               RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False
    bool NormalizedHammingInit(     RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False

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

def distance(s1, s2, *, processor=None, max=None):
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
    """
    cdef size_t c_max = <size_t>-1 if max is None else max
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return hamming_func(s1_proc.string, s2_proc.string, c_max)


def normalized_distance(s1, s2, *, processor=None, score_cutoff=None):
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
        If s1 and s2 have a different length

    See Also
    --------
    Hamming.distance : Hamming distance
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return normalized_hamming_func(s1_proc.string, s2_proc.string, c_score_cutoff)

cdef bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

cdef bool GetScorerFlagsHamming(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_U64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.u64 = 0
    dereference(scorer_flags).worst_score.u64 = SIZE_MAX
    return True

cdef bool GetScorerFlagsNormalizedHamming(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer HammingContext
HammingContext.version = SCORER_STRUCT_VERSION
HammingContext.kwargs_init = NoKwargsInit
HammingContext.get_scorer_flags = GetScorerFlagsHamming
HammingContext.scorer_func_init = HammingInit
distance._RF_Scorer = PyCapsule_New(&HammingContext, NULL, NULL)

cdef RF_Scorer NormalizedHammingContext
NormalizedHammingContext.version = SCORER_STRUCT_VERSION
NormalizedHammingContext.kwargs_init = NoKwargsInit
NormalizedHammingContext.get_scorer_flags = GetScorerFlagsNormalizedHamming
NormalizedHammingContext.scorer_func_init = NormalizedHammingInit
normalized_distance._RF_Scorer = PyCapsule_New(&NormalizedHammingContext, NULL, NULL)

