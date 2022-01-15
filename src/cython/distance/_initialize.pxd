from libcpp.vector cimport vector
from libcpp cimport bool
from cpp_common cimport RfEditops, RfOpcodes

cdef class Editops:
    cdef RfEditops editops

cdef class Opcodes:
    cdef RfOpcodes opcodes
