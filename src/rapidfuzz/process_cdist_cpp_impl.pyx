# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz.fuzz import ratio
from rapidfuzz.utils import default_process

from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp cimport bool
from libc.math cimport floor
from libc.stdint cimport uint32_t, uint64_t, int64_t

cimport cython
from cython.operator cimport dereference
from cpython cimport Py_buffer
from cpython.buffer cimport PyBUF_ND, PyBUF_SIMPLE, PyBUF_F_CONTIGUOUS

from cpp_common cimport (
    PyObjectWrapper, RF_StringWrapper, RF_KwargsWrapper,
    get_score_cutoff_f64, get_score_cutoff_i64,
    conv_sequence
)

from array import array

from rapidfuzz_capi cimport (
    RF_Preprocess, RF_Kwargs, RF_String, RF_Scorer, RF_ScorerFunc,
    RF_Preprocessor, RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_I64,
    RF_SCORER_FLAG_SYMMETRIC
)
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython.object cimport PyObject

cdef extern from "process_cdist_cpp.hpp":
    cpdef enum class MatrixType:
        UNDEFINED = 0
        FLOAT32 = 1
        FLOAT64 = 2
        INT8 = 3
        INT16 = 4
        INT32 = 5
        INT64 = 6
        UINT8 = 7
        UINT16 = 8
        UINT32 = 9
        UINT64 = 10

    cdef cppclass RfMatrix "Matrix":
        RfMatrix() except +
        RfMatrix(MatrixType, size_t, size_t) except +
        int get_dtype_size() except +
        const char* get_format() except +
        void set[T](size_t, size_t, T) except +

        MatrixType m_dtype
        size_t m_rows
        size_t m_cols
        void* m_matrix

    RfMatrix cdist_single_list_impl[T](  const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, MatrixType, int, T) except +
    RfMatrix cdist_two_lists_impl[T](    const RF_Kwargs*, RF_Scorer*,
        const vector[RF_StringWrapper]&, const vector[RF_StringWrapper]&, MatrixType, int, T) except +


FLOAT32 = MatrixType.FLOAT32
FLOAT64 = MatrixType.FLOAT64
INT8 = MatrixType.INT8
INT16 = MatrixType.INT16
INT32 = MatrixType.INT32
INT64 = MatrixType.INT64
UINT8 = MatrixType.UINT8
UINT16 = MatrixType.UINT16
UINT32 = MatrixType.UINT32
UINT64 = MatrixType.UINT64

cdef inline vector[PyObjectWrapper] preprocess_py(queries, processor) except *:
    cdef vector[PyObjectWrapper] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    proc_queries.reserve(queries_len)

    # processor None/False
    if not processor:
        for query in queries:
            proc_queries.emplace_back(<PyObject*>query)
    # processor has to be called through python
    else:
        for query in queries:
            proc_query = processor(query)
            proc_queries.emplace_back(<PyObject*>proc_query)

    return move(proc_queries)

cdef inline vector[RF_StringWrapper] preprocess(queries, processor) except *:
    cdef vector[RF_StringWrapper] proc_queries
    cdef int64_t queries_len = <int64_t>len(queries)
    cdef RF_String proc_str
    cdef RF_Preprocessor* processor_context = NULL
    proc_queries.reserve(queries_len)

    # No processor
    if not processor:
        for query in queries:
            proc_queries.emplace_back(conv_sequence(query))
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            processor_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        # use RapidFuzz C-Api
        if processor_context != NULL and processor_context.version == 1:
            for query in queries:
                processor_context.preprocess(query, &proc_str)
                proc_queries.emplace_back(proc_str)

        # Call Processor through Python
        else:
            for query in queries:
                proc_query = processor(query)
                proc_queries.emplace_back(conv_sequence(proc_query), <PyObject*>proc_query)

    return move(proc_queries)

cdef inline MatrixType dtype_to_type_num_f64(dtype) except MatrixType.UNDEFINED:
    if dtype is None:
        return MatrixType.FLOAT32
    return <MatrixType>dtype

cdef inline MatrixType dtype_to_type_num_i64(dtype) except MatrixType.UNDEFINED:
    if dtype is None:
        return MatrixType.INT32
    return <MatrixType>dtype

from cpython cimport Py_buffer
from libcpp.vector cimport vector

cdef class Matrix:
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef RfMatrix matrix

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self.shape[0] = self.matrix.m_rows
        self.shape[1] = self.matrix.m_cols
        self.strides[1] = self.matrix.get_dtype_size()
        self.strides[0] = self.matrix.m_cols * self.strides[1]

        buffer.buf = <char *>self.matrix.m_matrix
        buffer.format = <char *>self.matrix.get_format()
        buffer.internal = NULL
        buffer.itemsize = self.matrix.get_dtype_size()
        buffer.len = self.matrix.m_rows * self.matrix.m_cols * self.matrix.get_dtype_size()
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef cdist_two_lists(queries, choices, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    proc_choices = preprocess(choices, processor)
    flags = dereference(scorer_flags).flags
    cdef Matrix matrix = Matrix()

    if flags & RF_SCORER_FLAG_RESULT_F64:
        matrix.matrix = cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        matrix.matrix = cdist_two_lists_impl(
            kwargs, scorer, proc_queries, proc_choices,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))
    else:
        raise ValueError("scorer does not properly use the C-API")

    return matrix

cdef Matrix cdist_single_list(queries, RF_Scorer* scorer, const RF_ScorerFlags* scorer_flags, processor, score_cutoff, dtype, int c_workers, const RF_Kwargs* kwargs):
    proc_queries = preprocess(queries, processor)
    flags = dereference(scorer_flags).flags
    cdef Matrix matrix = Matrix()

    if flags & RF_SCORER_FLAG_RESULT_F64:
        matrix.matrix = cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_f64(dtype),
            c_workers,
            get_score_cutoff_f64(score_cutoff, scorer_flags))

    elif flags & RF_SCORER_FLAG_RESULT_I64:
        matrix.matrix = cdist_single_list_impl(
            kwargs, scorer, proc_queries,
            dtype_to_type_num_i64(dtype),
            c_workers,
            get_score_cutoff_i64(score_cutoff, scorer_flags))
    else:
        raise ValueError("scorer does not properly use the C-API")

    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cdist_py(queries, choices, scorer, processor, score_cutoff, dtype, workers, dict kwargs):
    proc_queries = preprocess_py(queries, processor)
    proc_choices = preprocess_py(choices, processor)
    cdef double score
    cdef Matrix matrix = Matrix()
    c_dtype = dtype_to_type_num_f64(dtype)
    matrix.matrix = RfMatrix(c_dtype, proc_queries.size(), proc_choices.size())

    kwargs["processor"] = None
    kwargs["score_cutoff"] = score_cutoff

    for i in range(proc_queries.size()):
        for j in range(proc_choices.size()):
            score = scorer(<object>proc_queries[i].obj, <object>proc_choices[j].obj,**kwargs)
            matrix.matrix.set(i, j, score)

    return matrix


def cdist(queries, choices, *, scorer=ratio, processor=None, score_cutoff=None, dtype=None, workers=1, **kwargs):
    cdef RF_Scorer* scorer_context = NULL
    cdef RF_ScorerFlags scorer_flags

    if processor is True:
        # todo: deprecate this
        processor = default_process
    elif processor is False:
        processor = None

    scorer_capsule = getattr(scorer, '_RF_Scorer', scorer)
    if PyCapsule_IsValid(scorer_capsule, NULL):
        scorer_context = <RF_Scorer*>PyCapsule_GetPointer(scorer_capsule, NULL)

    if scorer_context:
        if dereference(scorer_context).version == 1:
            kwargs_context = RF_KwargsWrapper()
            dereference(scorer_context).kwargs_init(&kwargs_context.kwargs, kwargs)
            dereference(scorer_context).get_scorer_flags(&kwargs_context.kwargs, &scorer_flags)

            # scorer(queries[i], choices[j]) == scorer(queries[j], choices[i])
            if scorer_flags.flags & RF_SCORER_FLAG_SYMMETRIC and queries is choices:
                return cdist_single_list(
                    queries, scorer_context, &scorer_flags, processor,
                    score_cutoff, dtype, workers, &kwargs_context.kwargs)
            else:
                return cdist_two_lists(
                    queries, choices, scorer_context, &scorer_flags, processor,
                    score_cutoff, dtype, workers, &kwargs_context.kwargs)

    return cdist_py(queries, choices, scorer, processor, score_cutoff, dtype, workers, kwargs)
