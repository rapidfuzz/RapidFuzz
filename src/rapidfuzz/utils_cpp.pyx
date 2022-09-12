# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from rapidfuzz_capi cimport RF_String, PREPROCESSOR_STRUCT_VERSION, RF_Preprocessor
from cpp_common cimport (
    is_valid_string, convert_string, hash_array, hash_sequence,
    conv_sequence
)

from cpython.pycapsule cimport PyCapsule_New
from libcpp cimport bool
from array import array

cdef extern from "utils_cpp.hpp":
    object default_process_impl(object) nogil except +
    void validate_string(object py_str, const char* err) except +
    RF_String default_process_func(RF_String sentence) except +

def default_process(sentence):
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

cdef RF_Preprocessor DefaultProcessContext
DefaultProcessContext.version = PREPROCESSOR_STRUCT_VERSION
DefaultProcessContext.preprocess = default_process_capi
default_process._RF_Preprocess = PyCapsule_New(&DefaultProcessContext, NULL, NULL)
