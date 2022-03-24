# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from libcpp.vector cimport vector
from libcpp cimport bool
from cpp_common cimport RfEditops, RfOpcodes

cdef class Editops:
    cdef RfEditops editops

cdef class Opcodes:
    cdef RfOpcodes opcodes

cdef class ScoreAlignment:
    cdef public object score
    cdef public Py_ssize_t src_start
    cdef public Py_ssize_t src_end
    cdef public Py_ssize_t dest_start
    cdef public Py_ssize_t dest_end