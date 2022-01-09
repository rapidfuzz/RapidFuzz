# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_U64, RF_SCORER_FLAG_MULTI_STRING, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence, vector_slice

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "<algorithm>" namespace "std" nogil:
    bool equal[InputIt1, InputIt2](InputIt1 first1, InputIt1 last1, InputIt2 first2, ...) except +

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

    cdef struct LevenshteinWeightTable:
        size_t insert_cost
        size_t delete_cost
        size_t replace_cost

cdef extern from "edit_based.hpp":
    ctypedef struct LevenshteinOpcode:
        LevenshteinEditType type
        size_t src_begin
        size_t src_end
        size_t dest_begin
        size_t dest_end

    vector[LevenshteinEditOp] opcodes_to_editops(const vector[LevenshteinOpcode]&) nogil except +
    vector[LevenshteinOpcode] editops_to_opcodes(const vector[LevenshteinEditOp]&, size_t, size_t) nogil except +


cdef str levenshtein_edit_type_to_str(LevenshteinEditType edit_type):
    if edit_type == LevenshteinEditType.Insert:
        return "insert"
    elif edit_type == LevenshteinEditType.Delete:
        return "delete"
    elif edit_type == LevenshteinEditType.Replace:
        return "replace"
    else:
        return "equal"

cdef list levenshtein_editops_to_list(vector[LevenshteinEditOp] ops):
    cdef size_t op_count = ops.size()
    cdef list result_list = PyList_New(<Py_ssize_t>op_count)
    for i in range(op_count):
        result_item = (levenshtein_edit_type_to_str(ops[i].type), ops[i].src_pos, ops[i].dest_pos)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list

cdef list levenshtein_opcodes_to_list(vector[LevenshteinOpcode] ops):
    cdef size_t op_count = ops.size()
    cdef list result_list = PyList_New(<Py_ssize_t>op_count)
    for i in range(op_count):
        result_item = (
            levenshtein_edit_type_to_str(ops[i].type),
            ops[i].src_begin, ops[i].src_end,
            ops[i].dest_begin, ops[i].dest_end)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list


cdef class Editops:
    """
    List like object of 3-tuples describing how to turn s1 into s2.
    Each tuple is of the form (tag, src_pos, dest_pos).

    The tags are strings, with these meanings:
    'replace':  s1[src_pos] should be replaced by s2[dest_pos]
    'delete':   s1[src_pos] should be deleted.
    'insert':   s2[dest_pos] should be inserted at s1[src_pos].
    """

    @classmethod
    def from_opcodes(cls, Opcodes opcodes):
        """
        Create Editops from Opcodes

        Parameters
        ----------
        opcodes : Opcodes
            opcodes to convert to editops

        Returns
        -------
        editops : Editops
            Opcodes converted to Editops
        """
        cdef Editops self = cls.__new__(cls)
        self.editops = opcodes_to_editops(opcodes.opcodes)
        self.len_s1 = opcodes.len_s1
        self.len_s2 = opcodes.len_s2
        return self

    def as_opcodes(self):
        """
        Convert to Opcodes

        Returns
        -------
        opcodes : Opcodes
            Editops converted to Opcodes
        """
        cdef Opcodes opcodes = Opcodes.__new__(Opcodes)
        opcodes.opcodes = editops_to_opcodes(self.editops, self.len_s1, self.len_s2)
        opcodes.len_s1 = self.len_s1
        opcodes.len_s2 = self.len_s2
        return opcodes

    def as_list(self):
        """
        Convert Editops to a list of tuples.

        This is the equivalent of ``[x for x in editops]``
        """
        return levenshtein_editops_to_list(self.editops)

    def __eq__(self, other):
        if isinstance(other, Editops):
            return equal(
                self.editops.begin(), self.editops.end(),
                (<Editops>other).editops.begin(), (<Editops>other).editops.end()
            )

        # todo implement comparision to list/Opcodes
        return False

    def __len__(self):
        return self.editops.size()

    def __getitem__(self, int index):
        if index < 0:
            index += self.editops.size()

        if index < 0 or index >= self.editops.size():
            raise IndexError("Editops index out of range")

        return (
            levenshtein_edit_type_to_str(self.editops[index].type),
            self.editops[index].src_pos,
            self.editops[index].dest_pos
        )

    def __repr__(self):
        return "[" + ", ".join(repr(op) for op in self) + "]"

cdef class Opcodes:
    """
    List like object of 5-tuples describing how to turn s1 into s2.
    Each tuple is of the form (tag, i1, i2, j1, j2). The first tuple
    has i1 == j1 == 0, and remaining tuples have i1 == the i2 from the
    tuple preceding it, and likewise for j1 == the previous j2.

    The tags are strings, with these meanings:
    'replace':  s1[i1:i2] should be replaced by s2[j1:j2]
    'delete':   s1[i1:i2] should be deleted.
                Note that j1==j2 in this case.
    'insert':   s2[j1:j2] should be inserted at s1[i1:i1].
                Note that i1==i2 in this case.
    'equal':    s1[i1:i2] == s2[j1:j2]

    Note
    --------
    Opcodes uses tuples similar to difflib's SequenceMatcher to make them
    interoperable
    """

    @classmethod
    def from_editops(cls, Editops editops):
        """
        Create Opcodes from Editops

        Parameters
        ----------
        editops : Editops
            editops to convert to opcodes

        Returns
        -------
        opcodes : Opcodes
            Editops converted to Opcodes
        """
        cdef Opcodes self = cls.__new__(cls)
        self.opcodes = editops_to_opcodes(editops.editops, editops.len_s1, editops.len_s2)
        self.len_s1 = editops.len_s1
        self.len_s2 = editops.len_s2
        return self

    def as_editops(self):
        """
        Convert Opcodes to Editops

        Returns
        -------
        editops : Editops
            Opcodes converted to Editops
        """
        cdef Editops editops = Editops.__new__(Editops)
        editops.editops = opcodes_to_editops(self.opcodes)
        editops.len_s1 = self.len_s1
        editops.len_s2 = self.len_s2
        return editops

    def as_list(self):
        """
        Convert Opcodes to a list of tuples, which is compatible
        with the opcodes of difflibs SequenceMatcher.

        This is the equivalent of ``[x for x in opcodes]``
        """
        return levenshtein_opcodes_to_list(self.opcodes)

    def __eq__(self, other):
        if isinstance(other, Opcodes):
            return equal(
                self.opcodes.begin(), self.opcodes.end(),
                (<Opcodes>other).opcodes.begin(), (<Opcodes>other).opcodes.end()
            )

        # todo implement comparision to list/Editops
        return False

    def __len__(self):
        return self.opcodes.size()

    def __getitem__(self, int index):
        if index < 0:
            index += self.opcodes.size()

        if index < 0 or index >= self.opcodes.size():
            raise IndexError("Opcodes index out of range")

        return (
            levenshtein_edit_type_to_str(self.opcodes[index].type),
            self.opcodes[index].src_begin,
            self.opcodes[index].src_end,
            self.opcodes[index].dest_begin,
            self.opcodes[index].dest_end
        )

    def __repr__(self):
        return "[" + ", ".join(repr(op) for op in self) + "]"
