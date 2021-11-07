# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport RF_String
from cpp_common cimport is_valid_string, convert_string, hash_array, hash_sequence

from cpython.pycapsule cimport PyCapsule_New
from libcpp cimport bool

cdef inline RF_String conv_sequence(seq) except *:
    if is_valid_string(seq):
        return convert_string(seq)
    elif isinstance(seq, array):
        return hash_array(seq)
    else:
        return hash_sequence(seq)

cdef extern from "cpp_utils.hpp":
    object default_process_impl(object) nogil except +
    void validate_string(object py_str, const char* err) except +
    RF_String default_process_func(RF_String sentence) except +

def default_process(sentence):
    """
    This function preprocesses a string by:
    - removing all non alphanumeric characters
    - trimming whitespaces
    - converting all characters to lower case

    Parameters
    ----------
    sentence : str
        String to preprocess

    Returns
    -------
    processed_string : str
        processed string
    """
    validate_string(sentence, "sentence must be a String")
    return default_process_impl(sentence)


cdef bool default_process_capi(sentence, RF_String* str_) except False:
    proc_str = conv_sequence(sentence)
    try:
        proc_str = default_process_func(proc_str)
    except:
        if proc_str.dtor:
            proc_str.dtor(&proc_str)
        raise
    
    str_[0] = proc_str
    return True

default_process._RF_Preprocess = PyCapsule_New(<void*>default_process_capi, NULL, NULL)