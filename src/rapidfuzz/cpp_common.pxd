# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from libc.stdint cimport uint64_t, int64_t
from libc.stdlib cimport malloc, free
from libc.stddef cimport wchar_t
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.utility cimport move
from cpython.object cimport PyObject
from cython.operator cimport dereference
from libcpp.vector cimport vector
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer

from rapidfuzz_capi cimport (
    RF_Scorer, RF_StringType, RF_String, RF_Kwargs, RF_KwargsInit,
    SCORER_STRUCT_VERSION, 
    RF_ScorerFlags, RF_Preprocessor, RF_GetScorerFlags, RF_ScorerFuncInit
)

from array import array

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cpdef enum class EditType:
        None    = 0,
        Replace = 1,
        Insert  = 2,
        Delete  = 3

    ctypedef struct RfEditOp "rapidfuzz::EditOp":
        EditType type
        size_t src_pos
        size_t dest_pos

    cdef cppclass RfScoreAlignment "rapidfuzz::ScoreAlignment"[T]:
        T score
        size_t src_start
        size_t src_end
        size_t dest_start
        size_t dest_end

        RfScoreAlignment()
        RfScoreAlignment(T score, size_t src_start, size_t src_end, size_t dest_start, size_t dest_end)

    cdef cppclass RfOpcodes "rapidfuzz::Opcodes"

    cdef cppclass RfEditops "rapidfuzz::Editops":
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            RfEditOp& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator++(int)
            iterator operator--(int)
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)
        cppclass const_iterator:
            const_iterator() except +
            const_iterator(iterator&) except +
            const_iterator(const_iterator&) except +
            operator=(iterator&) except +
            const RfEditOp& operator*()
            const_iterator operator++()
            const_iterator operator--()
            const_iterator operator++(int)
            const_iterator operator--(int)
            const_iterator operator+(size_type)
            const_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        RfEditops() except +
        RfEditops(size_t) except +
        RfEditops(const RfEditops&) except +
        RfEditops(const RfOpcodes&) except +
        bool operator==(const RfEditops&)
        RfEditOp& operator[](size_t pos) except +
        size_t size()
        RfEditops inverse() except +
        RfEditops remove_subsequence(const RfEditops& subsequence) except +
        RfEditops slice(int, int, int) except +
        void remove_slice(int, int, int) except +
        size_t get_src_len()
        void set_src_len(size_t)
        size_t get_dest_len()
        void set_dest_len(size_t)
        RfEditops reverse()
        void emplace_back(...)
        void reserve(size_t) except +
        void shrink_to_fit() except +

        iterator begin()
        iterator end()
        const_iterator cbegin()
        const_iterator cend()
        iterator erase(iterator)

    ctypedef struct RfOpcode "rapidfuzz::Opcode":
        EditType type
        size_t src_begin
        size_t src_end
        size_t dest_begin
        size_t dest_end

    cdef cppclass RfOpcodes "rapidfuzz::Opcodes":
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            RfOpcode& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator++(int)
            iterator operator--(int)
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)
        cppclass const_iterator:
            const_iterator() except +
            const_iterator(iterator&) except +
            const_iterator(const_iterator&) except +
            operator=(iterator&) except +
            const RfOpcode& operator*()
            const_iterator operator++()
            const_iterator operator--()
            const_iterator operator++(int)
            const_iterator operator--(int)
            const_iterator operator+(size_type)
            const_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        RfOpcodes() except +
        RfOpcodes(size_t) except +
        RfOpcodes(const RfOpcodes&) except +
        RfOpcodes(const RfEditops&) except +
        bool operator==(const RfOpcodes&)
        RfOpcode& operator[](size_t pos) except +
        size_t size()
        RfOpcodes inverse() except +
        size_t get_src_len()
        void set_src_len(size_t)
        size_t get_dest_len()
        void set_dest_len(size_t)
        RfOpcodes reverse()
        void emplace_back(...)
        void reserve(size_t) except +
        RfOpcode& back() except +
        void shrink_to_fit() except +
        bint empty()

        iterator begin()
        iterator end()
        const_iterator cbegin()
        const_iterator cend()
        iterator erase(iterator)

cdef extern from "cpp_common.hpp":
    cdef cppclass RF_StringWrapper:
        RF_String string
        PyObject* obj

        RF_StringWrapper()
        RF_StringWrapper(RF_String)
        RF_StringWrapper(RF_String, object)

    cdef cppclass RF_KwargsWrapper:
        RF_Kwargs kwargs

        RF_KwargsWrapper()
        RF_KwargsWrapper(RF_Kwargs)

    cdef cppclass PyObjectWrapper:
        PyObject* obj

        PyObjectWrapper()
        PyObjectWrapper(object)

    void default_string_deinit(RF_String* string) nogil

    int is_valid_string(object py_str) except +
    RF_String convert_string(object py_str)
    void validate_string(object py_str, const char* err) except +

cdef inline RF_String hash_array(arr) except *:
    # TODO on Cpython this does not require any copies
    cdef RF_String s_proc
    cdef Py_UCS4 typecode = <Py_UCS4>arr.typecode
    s_proc.length = <int64_t>len(arr)

    s_proc.data = malloc(s_proc.length * sizeof(uint64_t))

    if s_proc.data == NULL:
        raise MemoryError

    try:
        # ignore signed/unsigned, since it is not relevant in any of the algorithms
        if typecode in {'b', 'B'}: # signed/unsigned char
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode == 'u': # 'u' wchar_t
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t><Py_UCS4>arr[i]
        elif typecode in {'h', 'H'}: #  signed/unsigned short
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'i', 'I'}: # signed/unsigned int
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'l', 'L'}: # signed/unsigned long
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'q', 'Q'}: # signed/unsigned long long
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        else: # float/double are hashed
            s_proc.kind = RF_StringType.RF_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>hash(arr[i])
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.dtor = default_string_deinit
    return s_proc


cdef inline RF_String hash_sequence(seq) except *:
    cdef RF_String s_proc
    s_proc.length = <int64_t>len(seq)

    s_proc.data = malloc(s_proc.length * sizeof(uint64_t))

    if s_proc.data == NULL:
        raise MemoryError

    try:
        s_proc.kind = RF_StringType.RF_UINT64
        for i in range(s_proc.length):
            elem = seq[i]
            # this is required so e.g. a list of char can be compared to a string
            if isinstance(elem, str) and len(elem) == 1:
                (<uint64_t*>s_proc.data)[i] = <uint64_t><Py_UCS4>elem
            elif isinstance(elem, int) and elem == -1:
                (<uint64_t*>s_proc.data)[i] = <uint64_t>-1
            else:
                (<uint64_t*>s_proc.data)[i] = <uint64_t>hash(elem)
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.dtor = default_string_deinit
    return s_proc

cdef inline RF_String conv_sequence(seq) except *:
    if is_valid_string(seq):
        return move(convert_string(seq))
    elif isinstance(seq, array):
        return move(hash_array(seq))
    else:
        return move(hash_sequence(seq))

cdef inline double get_score_cutoff_f64(score_cutoff, const RF_ScorerFlags* scorer_flags) except *:
    worst_score = dereference(scorer_flags).worst_score.f64
    optimal_score = dereference(scorer_flags).optimal_score.f64
    c_score_cutoff = worst_score

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
        if optimal_score > worst_score:
            # e.g. 0.0 - 100.0
            if c_score_cutoff < worst_score or c_score_cutoff > optimal_score:
                raise TypeError(f"score_cutoff has to be in the range of {worst_score} - {optimal_score}")
        else:
            # e.g. DBL_MAX - 0
            if c_score_cutoff > worst_score or c_score_cutoff < optimal_score:
                raise TypeError(f"score_cutoff has to be in the range of {optimal_score} - {worst_score}")

    return c_score_cutoff

cdef inline int64_t get_score_cutoff_i64(score_cutoff, const RF_ScorerFlags* scorer_flags) except *:
    worst_score = dereference(scorer_flags).worst_score.i64
    optimal_score = dereference(scorer_flags).optimal_score.i64
    c_score_cutoff = worst_score

    if score_cutoff is not None:
        c_score_cutoff = score_cutoff
        if optimal_score > worst_score:
            # e.g. 0.0 - 100.0
            if c_score_cutoff < worst_score or c_score_cutoff > optimal_score:
                raise TypeError(f"score_cutoff has to be in the range of {worst_score} - {optimal_score}")
        else:
            # e.g. DBL_MAX - 0
            if c_score_cutoff > worst_score or c_score_cutoff < optimal_score:
                raise TypeError(f"score_cutoff has to be in the range of {optimal_score} - {worst_score}")

    return c_score_cutoff

cdef inline void preprocess_strings(s1, s2, processor, RF_StringWrapper* s1_proc, RF_StringWrapper* s2_proc, object default) except *:
    cdef RF_Preprocessor* preprocess_context = NULL

    if processor is True:
        # todo: deprecate
        processor = default

    if not processor:
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

cdef inline bool NoKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    if len(kwargs):
        raise TypeError("Got unexpected keyword arguments: ", ", ".join(kwargs.keys()))

    dereference(self).context = NULL
    dereference(self).dtor = NULL
    return True

cdef inline RF_Scorer CreateScorerContext(RF_KwargsInit kwargs_init, RF_GetScorerFlags get_scorer_flags, RF_ScorerFuncInit scorer_func_init):
    cdef RF_Scorer context
    context.version = SCORER_STRUCT_VERSION
    context.kwargs_init = kwargs_init
    context.get_scorer_flags = get_scorer_flags
    context.scorer_func_init = scorer_func_init
    return context

cdef inline dict CreateScorerContextPy(get_scorer_flags):
    return {"get_scorer_flags": get_scorer_flags}

cdef inline bool AddScorerContext(func, py_context, RF_Scorer* c_context) except False:
    func._RF_Scorer = PyCapsule_New(c_context, NULL, NULL)
    func._RF_ScorerPy = py_context
    return True
