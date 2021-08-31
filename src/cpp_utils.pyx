# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

cdef extern from "cpp_common.hpp":
    void validate_string(object py_str, const char* err) except +

cdef extern from "cpp_utils.hpp":
    object default_process_impl(object) nogil except +

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