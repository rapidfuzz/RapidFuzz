# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence
from libc.stdint cimport SIZE_MAX

from libcpp cimport bool
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "edit_based.hpp":
    double jaro_winkler_similarity_func(const RF_String&, const RF_String&, double, double) nogil except +
    bool JaroWinklerSimilarityInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

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

def similarity(s1, s2, *, double prefix_weight=0.1, processor=None, score_cutoff=None):
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
        If prefix_weight is invalid
    """
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return jaro_winkler_similarity_func(s1_proc.string, s2_proc.string, prefix_weight, c_score_cutoff)

cdef void KwargsDeinit(RF_Kwargs* self):
    free(<void*>dereference(self).context)

cdef bool JaroWinklerKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    cdef double* prefix_weight = <double*>malloc(sizeof(double))

    if not prefix_weight:
        raise MemoryError

    prefix_weight[0] = kwargs.get("prefix_weight", 0.1)
    dereference(self).context = prefix_weight
    dereference(self).dtor = KwargsDeinit
    return True

cdef bool GetScorerFlagsJaroWinklerSimilarity(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer JaroWinklerSimilarityContext
JaroWinklerSimilarityContext.version = SCORER_STRUCT_VERSION
JaroWinklerSimilarityContext.kwargs_init = JaroWinklerKwargsInit
JaroWinklerSimilarityContext.get_scorer_flags = GetScorerFlagsJaroWinklerSimilarity
JaroWinklerSimilarityContext.scorer_func_init = JaroWinklerSimilarityInit
similarity._RF_Scorer = PyCapsule_New(&JaroWinklerSimilarityContext, NULL, NULL)
