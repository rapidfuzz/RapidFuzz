from libcpp.vector cimport vector

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cpdef enum class LevenshteinEditType:
        None    = 0,
        Replace = 1,
        Insert  = 2,
        Delete  = 3

    ctypedef struct LevenshteinEditOp:
        LevenshteinEditType type
        size_t src_pos
        size_t dest_pos

cdef extern from "edit_based.hpp":
    ctypedef struct LevenshteinOpcode:
        LevenshteinEditType type
        size_t src_begin
        size_t src_end
        size_t dest_begin
        size_t dest_end

cdef class Editops:
    cdef:
        vector[LevenshteinEditOp] editops
        size_t len_s1
        size_t len_s2

cdef class Opcodes:
    cdef:
        vector[LevenshteinOpcode] opcodes
        size_t len_s1
        size_t len_s2
