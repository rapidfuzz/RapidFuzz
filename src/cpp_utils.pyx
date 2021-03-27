# distutils: language=c++
# cython: language_level=3
# cython: binding=True

cdef extern from "cpp_utils.hpp":
    object default_process_impl(object) except +

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
    return default_process_impl(sentence)