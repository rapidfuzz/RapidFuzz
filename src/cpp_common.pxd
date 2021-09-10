from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.stddef cimport wchar_t
from libcpp.utility cimport move
from libcpp cimport bool

cdef extern from "cpp_common.hpp":
    ctypedef unsigned int RapidfuzzType
    int RAPIDFUZZ_UINT8
    int RAPIDFUZZ_UINT16
    int RAPIDFUZZ_UINT32
    int RAPIDFUZZ_UINT64
    int RAPIDFUZZ_INT64

    cdef cppclass proc_string:
        RapidfuzzType kind
        bool allocated
        void* data
        size_t length

        proc_string()

    int is_valid_string(object py_str) except +
    proc_string convert_string(object py_str)
    void validate_string(object py_str, const char* err) except +
    proc_string default_process_func(proc_string sentence) except +

cdef inline proc_string hash_array(arr) except *:
    # TODO on Cpython this does not require any copies
    cdef proc_string s_proc
    cdef Py_UCS4 typecode = <Py_UCS4>arr.typecode
    s_proc.length = <size_t>len(arr)

    s_proc.data = malloc(s_proc.length * sizeof(uint64_t))

    if s_proc.data == NULL:
        raise MemoryError

    try:
        # ignore signed/unsigned, since it is not relevant in any of the algorithms
        if typecode in {'b', 'B'}: # signed/unsigned char
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode == 'u': # 'u' wchar_t
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t><Py_UCS4>arr[i]
        elif typecode in {'h', 'H'}: #  signed/unsigned short
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'i', 'I'}: # signed/unsigned int
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'l', 'L'}: # signed/unsigned long
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        elif typecode in {'q', 'Q'}: # signed/unsigned long long
            s_proc.kind = RAPIDFUZZ_UINT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>arr[i]
        else: # float/double are hashed
            s_proc.kind = RAPIDFUZZ_INT64
            for i in range(s_proc.length):
                (<uint64_t*>s_proc.data)[i] = <uint64_t>hash(arr[i])
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.allocated = True
    return move(s_proc)


cdef inline proc_string hash_sequence(seq) except *:
    cdef proc_string s_proc
    s_proc.length = <size_t>len(seq)

    s_proc.data = malloc(s_proc.length * sizeof(uint64_t))

    if s_proc.data == NULL:
        raise MemoryError

    try:
        s_proc.kind = RAPIDFUZZ_INT64
        for i in range(s_proc.length):
            elem = seq[i]
            # this is required so e.g. a list of char can be compared to a string
            if isinstance(elem, str) and len(elem) == 1:
                (<uint64_t*>s_proc.data)[i] = <uint64_t><Py_UCS4>elem
            else:
                (<uint64_t*>s_proc.data)[i] = <uint64_t>hash(elem)
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.allocated = True
    return move(s_proc)
