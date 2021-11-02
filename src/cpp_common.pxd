from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free
from libc.stddef cimport wchar_t
from libcpp cimport bool

from rapidfuzz_capi cimport RF_StringType, RF_String, RF_Kwargs

cdef extern from "cpp_common.hpp":
    cdef cppclass RF_StringWrapper:
        RF_String string
        
        RF_StringWrapper()
        RF_StringWrapper(RF_String)
    
    cdef cppclass RF_KwargsWrapper:
        RF_Kwargs kwargs
        
        RF_KwargsWrapper()
        RF_KwargsWrapper(RF_Kwargs)

    void default_string_deinit(RF_String* string)

    int is_valid_string(object py_str) except +
    RF_String convert_string(object py_str)
    void validate_string(object py_str, const char* err) except +
    RF_String default_process_func(RF_String sentence) except +

cdef inline RF_String hash_array(arr) except *:
    # TODO on Cpython this does not require any copies
    cdef RF_String s_proc
    cdef Py_UCS4 typecode = <Py_UCS4>arr.typecode
    s_proc.length = <size_t>len(arr)

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

    s_proc.deinit = default_string_deinit
    return s_proc


cdef inline RF_String hash_sequence(seq) except *:
    cdef RF_String s_proc
    s_proc.length = <size_t>len(seq)

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
            else:
                (<uint64_t*>s_proc.data)[i] = <uint64_t>hash(elem)
    except Exception as e:
        free(s_proc.data)
        s_proc.data = NULL
        raise

    s_proc.deinit = default_string_deinit
    return s_proc
