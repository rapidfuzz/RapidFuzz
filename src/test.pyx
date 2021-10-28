cimport cython
from cython.parallel import prange

cdef extern from *:
    """
    class Test {
    };
    """
    cdef cppclass Test:
        Test() nogil except +

@cython.cpp_locals(True)
cdef test():
    cdef int i
    for i in prange(10, nogil=True):
        var = Test()