# distutils: language=c++
# cython: language_level=3
# cython: binding=True

cdef extern from "cpp_utils.hpp":
    object default_process_impl(object) except +*


cdef dummy() except +:
    # trick cython into generating
    # exception handling, since except +* does not work properly
    # https://github.com/cython/cython/issues/3065
    dummy()


def default_process(sentence):
    """
    This function preprocesses a string by:
    - removing all non alphanumeric characters
    - trimming whitespaces
    - converting all characters to lower case
    
    Right now this only affects characters lower than 256
    (extended Ascii), while all other characters are not modified.
    This should be enough for most western languages. Full Unicode
    support will be added in a later release.
    
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