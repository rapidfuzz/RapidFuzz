from libc.stdint cimport int64_t, uint8_t
from libc.stdlib cimport malloc, free
from libc.stddef cimport wchar_t

cdef extern from "cpp_common.hpp":
    ctypedef struct proc_string:
        uint8_t kind
        uint8_t allocated
        void* data
        size_t length

    int is_valid_string(object py_str) except +
    proc_string convert_string(object py_str)
    void validate_string(object py_str, const char* err) except +

    int RAPIDFUZZ_HASH
    int RAPIDFUZZ_CHAR
    int RAPIDFUZZ_UNSIGNED_CHAR
    int RAPIDFUZZ_WCHAR
    int RAPIDFUZZ_SHORT
    int RAPIDFUZZ_UNSIGNED_SHORT
    int RAPIDFUZZ_INT
    int RAPIDFUZZ_UNSIGNED_INT
    int RAPIDFUZZ_LONG
    int RAPIDFUZZ_UNSIGNED_LONG
    int RAPIDFUZZ_LONG_LONG
    int RAPIDFUZZ_UNSIGNED_LONG_LONG


cdef inline proc_string hash_array(arr):
    # TODO on Cpython this does not require any copies
    cdef proc_string s_proc
    cdef Py_UCS4 typecode = <Py_UCS4>arr.typecode
    s_proc.length = len(arr)

    if typecode in {'f', 'd'}: # float/double are hashed
        s_proc.data = malloc(s_proc.length * sizeof(Py_hash_t))
    else:
        s_proc.data = malloc(s_proc.length * arr.itemsize)

    if s_proc.data == NULL:
        raise MemoryError

    try:
        # ignore signed/unsigned, since it is not relevant in any of the algorithms
        if typecode in {'b', 'B'}: # signed/unsigned char
            s_proc.kind = RAPIDFUZZ_UNSIGNED_CHAR
            for i in range(s_proc.length):
                (<unsigned char*>s_proc.data)[i] = arr[i]
        elif typecode == 'u': # 'u' wchar_t
            s_proc.kind = RAPIDFUZZ_WCHAR
            for i in range(s_proc.length):
                (<wchar_t*>s_proc.data)[i] = <wchar_t><Py_UCS4>arr[i]
        elif typecode in {'h', 'H'}: #  signed/unsigned short
            s_proc.kind = RAPIDFUZZ_UNSIGNED_SHORT
            for i in range(s_proc.length):
                (<unsigned short*>s_proc.data)[i] = arr[i]
        elif typecode in {'i', 'I'}: # signed/unsigned int
            s_proc.kind = RAPIDFUZZ_UNSIGNED_INT
            for i in range(s_proc.length):
                (<unsigned int*>s_proc.data)[i] = arr[i]
        elif typecode in {'l', 'L'}: # signed/unsigned long
            s_proc.kind = RAPIDFUZZ_UNSIGNED_LONG
            for i in range(s_proc.length):
                (<unsigned long*>s_proc.data)[i] = arr[i]
        elif typecode in {'q', 'Q'}: # signed/unsigned long long
            s_proc.kind = RAPIDFUZZ_UNSIGNED_LONG_LONG
            for i in range(s_proc.length):
                (<unsigned long long*>s_proc.data)[i] = arr[i]
        else: # float/double are hashed
            s_proc.kind = RAPIDFUZZ_HASH
            for i in range(s_proc.length):
                (<Py_hash_t*>s_proc.data)[i] = hash(arr[i])
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.allocated = True
    return s_proc


cdef inline proc_string hash_sequence(seq):
    cdef proc_string s_proc
    s_proc.length = len(seq)

    s_proc.data = malloc(s_proc.length * sizeof(Py_hash_t))

    if s_proc.data == NULL:
        raise MemoryError

    try:
        s_proc.kind = RAPIDFUZZ_HASH
        for i in range(s_proc.length):
            elem = seq[i]
            # this is required so e.g. a list of char can be compared to a string
            if isinstance(elem, str) and len(elem) == 1:
                (<Py_hash_t*>s_proc.data)[i] = <Py_hash_t><Py_UCS4>elem
            else:
                (<Py_hash_t*>s_proc.data)[i] = hash(elem)
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.allocated = True
    return s_proc

"""
cdef inline proc_string conv_sequence(seq):
    if is_valid_string(seq):
        return convert_string(seq)
    elif seq is array:
        print("test")
        return hash_array(seq)
    else:
        return hash_sequence(seq)"""
