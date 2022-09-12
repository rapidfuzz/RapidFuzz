# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from .distance._initialize_cpp import ScoreAlignment

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_ScorerFunc, RF_Kwargs, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_SYMMETRIC
)

# required for preprocess_strings
from rapidfuzz.utils import default_process
from array import array
from cpp_common cimport (
    RF_StringWrapper, preprocess_strings, RfScoreAlignment, NoKwargsInit,
    CreateScorerContext, CreateScorerContextPy, AddScorerContext
)

from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cython.operator cimport dereference

from array import array

cdef extern from "fuzz_cpp.hpp":
    double ratio_func(                    const RF_String&, const RF_String&, double) nogil except +
    double partial_ratio_func(            const RF_String&, const RF_String&, double) nogil except +
    double token_sort_ratio_func(         const RF_String&, const RF_String&, double) nogil except +
    double token_set_ratio_func(          const RF_String&, const RF_String&, double) nogil except +
    double token_ratio_func(              const RF_String&, const RF_String&, double) nogil except +
    double partial_token_sort_ratio_func( const RF_String&, const RF_String&, double) nogil except +
    double partial_token_set_ratio_func(  const RF_String&, const RF_String&, double) nogil except +
    double partial_token_ratio_func(      const RF_String&, const RF_String&, double) nogil except +
    double WRatio_func(                   const RF_String&, const RF_String&, double) nogil except +
    double QRatio_func(                   const RF_String&, const RF_String&, double) nogil except +

    RfScoreAlignment[double] partial_ratio_alignment_func(const RF_String&, const RF_String&, double) nogil except +

    bool RatioInit(                 RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialRatioInit(          RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenSortRatioInit(        RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenSetRatioInit(         RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool TokenRatioInit(            RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenSortRatioInit( RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenSetRatioInit(  RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool PartialTokenRatioInit(     RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool WRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False
    bool QRatioInit(                RF_ScorerFunc*, const RF_Kwargs*, int64_t, const RF_String*) nogil except False

def ratio(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_ratio_alignment(s1, s2, *, processor=None, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
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

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_sort_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_set_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def partial_token_ratio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return partial_token_ratio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


def WRatio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return WRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)

def QRatio(s1, s2, *, processor=default_process, score_cutoff=None):
    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    cdef RF_StringWrapper s1_proc, s2_proc

    if s1 is None or s2 is None:
        return 0

    if processor is True:
        processor = default_process

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc, default_process)
    return QRatio_func(s1_proc.string, s2_proc.string, c_score_cutoff)


cdef bool GetScorerFlagsFuzzSymmetric(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64 | RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef bool GetScorerFlagsFuzz(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    dereference(scorer_flags).optimal_score.f64 = 100
    dereference(scorer_flags).worst_score.f64 = 0
    return True

def _GetScorerFlagsSimilarity(**kwargs):
    return {"optimal_score": 100, "worst_score": 0, "flags": (1 << 5)}

cdef dict FuzzContextPy = CreateScorerContextPy(_GetScorerFlagsSimilarity)

cdef RF_Scorer RatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, RatioInit)
AddScorerContext(ratio, FuzzContextPy, &RatioContext)

cdef RF_Scorer PartialRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialRatioInit)
AddScorerContext(partial_ratio, FuzzContextPy, &PartialRatioContext)

cdef RF_Scorer TokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenSortRatioInit)
AddScorerContext(token_sort_ratio, FuzzContextPy, &TokenSortRatioContext)

cdef RF_Scorer TokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenSetRatioInit)
AddScorerContext(token_set_ratio, FuzzContextPy, &TokenSetRatioContext)

cdef RF_Scorer TokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, TokenRatioInit)
AddScorerContext(token_ratio, FuzzContextPy, &TokenRatioContext)

cdef RF_Scorer PartialTokenSortRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSortRatioInit)
AddScorerContext(partial_token_sort_ratio, FuzzContextPy, &PartialTokenSortRatioContext)

cdef RF_Scorer PartialTokenSetRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenSetRatioInit)
AddScorerContext(partial_token_set_ratio, FuzzContextPy, &PartialTokenSetRatioContext)

cdef RF_Scorer PartialTokenRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, PartialTokenRatioInit)
AddScorerContext(partial_token_ratio, FuzzContextPy, &PartialTokenRatioContext)

cdef RF_Scorer WRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzz, WRatioInit)
AddScorerContext(WRatio, FuzzContextPy, &WRatioContext)

cdef RF_Scorer QRatioContext = CreateScorerContext(NoKwargsInit, GetScorerFlagsFuzzSymmetric, QRatioInit)
AddScorerContext(QRatio, FuzzContextPy, &QRatioContext)
