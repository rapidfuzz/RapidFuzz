# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from .distance._initialize_cpp import ScoreAlignment

from rapidfuzz cimport (
    RF_SCORER_FLAG_MULTI_STRING_INIT,
    RF_SCORER_FLAG_RESULT_F64,
    RF_SCORER_FLAG_SYMMETRIC,
    RF_Kwargs,
    RF_Scorer,
    RF_ScorerFlags,
    RF_ScorerFunc,
    RF_String,
)

# required for preprocess_strings

from array import array

from rapidfuzz.utils import default_process

from cpp_common cimport (
    CreateScorerContext,
    CreateScorerContextPy,
    NoKwargsInit,
    RF_StringWrapper,
    RfScoreAlignment,
    preprocess_strings,
)
from cpython.pycapsule cimport PyCapsule_New
from libc.math cimport isnan
from libc.stdint cimport int64_t, uint32_t
from libcpp cimport bool

from array import array


cdef extern from "fuzz_cpp.hpp":
    double ratio_func(                    const RF_String&, const RF_String&, double) except + nogil
    double partial_ratio_func(            const RF_String&, const RF_String&, double) except + nogil
    double token_sort_ratio_func(         const RF_String&, const RF_String&, double) except + nogil
    double token_set_ratio_func(          const RF_String&, const RF_String&, double) except + nogil
    double token_ratio_func(              const RF_String&, const RF_String&, double) except + nogil
    double partial_token_sort_ratio_func( const RF_String&, const RF_String&, double) except + nogil
    double partial_token_set_ratio_func(  const RF_String&, const RF_String&, double) except + nogil
    double partial_token_ratio_func(      const RF_String&, const RF_String&, double) except + nogil
    double WRatio_func(                   const RF_String&, const RF_String&, double) except + nogil
    double QRatio_func(                   const RF_String&, const RF_String&, double) except + nogil

    RfScoreAlignment[double] partial_ratio_alignment_func(const RF_String&, const RF_String&, double) except + nogil

    bool RatioInit(                 RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool PartialRatioInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool TokenSortRatioInit(        RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool TokenSetRatioInit(         RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool TokenRatioInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool PartialTokenSortRatioInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool PartialTokenSetRatioInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool PartialTokenRatioInit(     RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool WRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil
    bool QRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) except False nogil

    bool RatioMultiStringSupport(const RF_Kwargs*) nogil

cdef inline bool is_none(s):
    if s is None:
        return True

    if isinstance(s, float) and isnan(s):
        return True

    return False

def ratio(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio_alignment(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return None

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    res = partial_ratio_alignment_func(s1_proc.string, s2_proc.string, c_score_cutoff)

    if res.score >= c_score_cutoff:
        return ScoreAlignment(res.score, res.src_start, res.src_end, res.dest_start, res.dest_end)
    else:
        return None


def token_sort_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def WRatio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return WRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def QRatio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if is_none(s1) or is_none(s2):
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return QRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool GetScorerFlagsFuzz(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) except False nogil:
    scorer_flags.flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    scorer_flags.optimal_score.f64 = 100
    scorer_flags.worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsFuzzRatio(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) except False nogil:
    scorer_flags.flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    if RatioMultiStringSupport(self):
        scorer_flags.flags |= RF_SCORER_FLAG_MULTI_STRING_INIT

    scorer_flags.optimal_score.f64 = 100
    scorer_flags.worst_score.f64 = 0
    return True

cdef RF_Scorer RatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzRatio, RatioInit)
ratio._RF_Scorer = PyCapsule_New(&RatioContext, NULL, NULL)

cdef RF_Scorer PartialRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialRatioInit)
partial_ratio._RF_Scorer = PyCapsule_New(&PartialRatioContext, NULL, NULL)

cdef RF_Scorer TokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzRatio, TokenSortRatioInit)
token_sort_ratio._RF_Scorer = PyCapsule_New(&TokenSortRatioContext, NULL, NULL)

cdef RF_Scorer TokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, TokenSetRatioInit)
token_set_ratio._RF_Scorer = PyCapsule_New(&TokenSetRatioContext, NULL, NULL)

cdef RF_Scorer TokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, TokenRatioInit)
token_ratio._RF_Scorer = PyCapsule_New(&TokenRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSortRatioInit)
partial_token_sort_ratio._RF_Scorer = PyCapsule_New(&PartialTokenSortRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSetRatioInit)
partial_token_set_ratio._RF_Scorer = PyCapsule_New(&PartialTokenSetRatioContext, NULL, NULL)

cdef RF_Scorer PartialTokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenRatioInit)
partial_token_ratio._RF_Scorer = PyCapsule_New(&PartialTokenRatioContext, NULL, NULL)

cdef RF_Scorer WRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, WRatioInit)
WRatio._RF_Scorer = PyCapsule_New(&WRatioContext, NULL, NULL)

cdef RF_Scorer QRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzRatio, QRatioInit)
QRatio._RF_Scorer = PyCapsule_New(&QRatioContext, NULL, NULL)
