# distutils: language=c++
# cython: language_level=3, binding=True, linetrace=True

from array import array

from rapidfuzz_capi cimport (
    RF_String, RF_Scorer, RF_Kwargs, RF_ScorerFunc, RF_Preprocess, RF_KwargsInit,
    SCORER_STRUCT_VERSION, RF_Preprocessor,
    RF_ScorerFlags,
    RF_SCORER_FLAG_RESULT_F64, RF_SCORER_FLAG_RESULT_U64, RF_SCORER_FLAG_MULTI_STRING, RF_SCORER_FLAG_SYMMETRIC
)
from cpp_common cimport RF_StringWrapper, conv_sequence, vector_slice, RfEditOp, RfOpcode, EditType

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cython.operator cimport dereference

cdef extern from "rapidfuzz/details/types.hpp" namespace "rapidfuzz" nogil:
    cdef struct LevenshteinWeightTable:
        size_t insert_cost
        size_t delete_cost
        size_t replace_cost

cdef str levenshtein_edit_type_to_str(EditType edit_type):
    if edit_type == EditType.Insert:
        return "insert"
    elif edit_type == EditType.Delete:
        return "delete"
    elif edit_type == EditType.Replace:
        return "replace"
    else:
        return "equal"

cdef EditType levenshtein_str_to_edit_type(edit_type) except *:
    if edit_type == "insert":
        return EditType.Insert
    elif edit_type == "delete":
        return EditType.Delete
    elif edit_type == "replace":
        return EditType.Replace
    elif edit_type == "equal":
        return EditType.None
    else:
        raise ValueError("Invalid Edit Type")


cdef list levenshtein_editops_to_list(const RfEditops& ops):
    cdef size_t op_count = ops.size()
    cdef list result_list = PyList_New(<Py_ssize_t>op_count)
    for i in range(op_count):
        result_item = (levenshtein_edit_type_to_str(ops[i].type), ops[i].src_pos, ops[i].dest_pos)
        Py_INCREF(result_item)
        PyList_SET_ITEM(result_list, <Py_ssize_t>i, result_item)

    return result_list

cdef list levenshtein_opcodes_to_list(const RfOpcodes& ops):
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
        self.editops = RfEditops(opcodes.opcodes)
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
        opcodes.opcodes = RfOpcodes(self.editops)
        return opcodes

    def as_list(self):
        """
        Convert Editops to a list of tuples.

        This is the equivalent of ``[x for x in editops]``
        """
        return levenshtein_editops_to_list(self.editops)

    def copy(self):
        """
        performs copy of Editops
        """
        cdef Editops x = Editops.__new__(Editops)
        x.editops = self.editops
        return x

    def inverse(self):
        """
        Invert Editops, so it describes how to transform the destination string to
        the source string.

        Returns
        -------
        editops : Editops
            inverted Editops

        Examples
        --------
        >>> from rapidfuzz.distance import Levenshtein
        >>> Levenshtein.editops('spam', 'park')
        [('delete', 0, 0), ('replace', 3, 2), ('insert', 4, 3)]
        >>> Levenshtein.editops('spam', 'park').inverse()
        [('insert', 0, 0), ('replace', 2, 3), ('delete', 3, 4)]
        """
        cdef Editops x = Editops.__new__(Editops)
        x.editops = self.editops.inverse()
        return x

    @property
    def src_len(self):
        return self.editops.get_src_len()

    @src_len.setter
    def src_len(self, value):
        self.editops.set_src_len(value)

    @property
    def dest_len(self):
        return self.editops.get_dest_len()

    @dest_len.setter
    def dest_len(self, value):
        self.editops.set_dest_len(value)

    def __eq__(self, other):
        if isinstance(other, Editops):
            return self.editops == (<Editops>other).editops

        return False

    def __len__(self):
        return self.editops.size()

    def __setitem__(self, int index, tuple value):
        cdef size_t src_pos, dest_pos

        if index < 0:
            index += self.editops.size()

        if index < 0 or index >= self.editops.size():
            raise IndexError("Editops index out of range")

        edit_type, src_pos, dest_pos = value

        self.editops[index] = RfEditOp(
            levenshtein_str_to_edit_type(edit_type),
            src_pos, dest_pos
        )

    def __getitem__(self, key):
        cdef int index, start, stop, step
        cdef Editops x

        if isinstance(key, slice):
            start = <slice>key.start if <slice>key.start is not None else 0
            stop = <slice>key.stop if <slice>key.stop is not None else self.editops.size()
            step = <slice>key.step if <slice>key.step is not None else 1
            x = Editops.__new__(Editops)
            x.editops = self.editops.slice(start, stop, step)
            return x
        elif isinstance(key, int):
            index = key
            if index < 0:
                index += self.editops.size()

            if index < 0 or index >= self.editops.size():
                raise IndexError("Editops index out of range")

            return (
                levenshtein_edit_type_to_str(self.editops[index].type),
                self.editops[index].src_pos,
                self.editops[index].dest_pos
            )
        else:
            raise TypeError("Expected slice or index")

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
        self.opcodes = RfOpcodes(editops.editops)
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
        editops.editops = RfEditops(self.opcodes)
        return editops

    def as_list(self):
        """
        Convert Opcodes to a list of tuples, which is compatible
        with the opcodes of difflibs SequenceMatcher.

        This is the equivalent of ``[x for x in opcodes]``
        """
        return levenshtein_opcodes_to_list(self.opcodes)

    def copy(self):
        """
        performs copy of Opcodes
        """
        cdef Opcodes x = Opcodes.__new__(Opcodes)
        x.opcodes = self.opcodes
        return x

    def inverse(self):
        """
        Invert Opcodes, so it describes how to transform the destination string to
        the source string.

        Returns
        -------
        opcodes : Opcodes
            inverted Opcodes

        Examples
        --------
        >>> from rapidfuzz.distance import Levenshtein
        >>> Levenshtein.opcodes('spam', 'park')
        [('delete', 0, 1, 0, 0), ('equal', 1, 3, 0, 2), ('replace', 3, 4, 2, 3),
         ('insert', 4, 4, 3, 4)]
        >>> Levenshtein.opcodes('spam', 'park').inverse()
        [('insert', 0, 0, 0, 1), ('equal', 0, 2, 1, 3), ('replace', 2, 3, 3, 4),
         ('delete', 3, 4, 4, 4)]
        """
        cdef Opcodes x = Opcodes.__new__(Opcodes)
        x.opcodes = self.opcodes.inverse()
        return x

    @property
    def src_len(self):
        return self.editops.get_src_len()

    @src_len.setter
    def src_len(self, value):
        self.editops.set_src_len(value)

    @property
    def dest_len(self):
        return self.editops.get_dest_len()

    @dest_len.setter
    def dest_len(self, value):
        self.editops.set_dest_len(value)

    def __eq__(self, other):
        if isinstance(other, Opcodes):
            return self.opcodes == (<Opcodes>other).opcodes

        return False

    def __len__(self):
        return self.opcodes.size()

    def __setitem__(self, int index, tuple value):
        cdef size_t src_begin, src_end, dest_begin, dest_end

        if index < 0:
            index += self.opcodes.size()

        if index < 0 or index >= self.opcodes.size():
            raise IndexError("Opcodes index out of range")

        edit_type, src_begin, src_end, dest_begin, dest_end = value

        self.opcodes[index] = RfOpcode(
            levenshtein_str_to_edit_type(edit_type),
            src_begin, src_end, dest_begin, dest_end
        )

    def __getitem__(self, key):
        cdef int index, start, stop, step
        cdef Opcodes x

        if isinstance(key, slice):
            start = <slice>key.start if <slice>key.start is not None else 0
            stop = <slice>key.stop if <slice>key.stop is not None else self.opcodes.size()
            step = <slice>key.step if <slice>key.step is not None else 1
            x = Opcodes.__new__(Opcodes)
            x.opcodes = self.opcodes.slice(start, stop, step)
            return x
        elif isinstance(key, int):
            index = key
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
        else:
            raise TypeError("Expected slice or index")

    def __repr__(self):
        return "[" + ", ".join(repr(op) for op in self) + "]"
