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
from libc.stdint cimport SIZE_MAX

from libcpp cimport bool
from libcpp.vector cimport vector
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
    ctypedef struct LevenshteinOpcodes:
        LevenshteinEditType type
        size_t src_begin
        size_t src_end
        size_t dest_begin
        size_t dest_end

    double normalized_levenshtein_func( const RF_String&, const RF_String&, size_t, size_t, size_t, double) nogil except +

    size_t levenshtein_func(const RF_String&, const RF_String&, size_t, size_t, size_t, size_t) nogil except +

    vector[LevenshteinEditOp] levenshtein_editops_func(const RF_String&, const RF_String&) nogil except +
    vector[LevenshteinEditOp] opcodes_to_editops(const vector[LevenshteinOpcodes]&) nogil except +
    vector[LevenshteinOpcodes] editops_to_opcodes(const vector[LevenshteinEditOp]&, size_t, size_t) nogil except +

    bool LevenshteinInit(           RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False
    bool NormalizedLevenshteinInit( RF_ScorerFunc*, const RF_Kwargs*, size_t, const RF_String*) nogil except False

cdef inline void preprocess_strings(s1, s2, processor, RF_StringWrapper* s1_proc, RF_StringWrapper* s2_proc) except *:
    cdef RF_Preprocessor* preprocess_context = NULL

    if processor is None:
        s1_proc[0] = RF_StringWrapper(conv_sequence(s1))
        s2_proc[0] = RF_StringWrapper(conv_sequence(s2))
    else:
        processor_capsule = getattr(processor, '_RF_Preprocess', processor)
        if PyCapsule_IsValid(processor_capsule, NULL):
            preprocess_context = <RF_Preprocessor*>PyCapsule_GetPointer(processor_capsule, NULL)

        if preprocess_context != NULL and preprocess_context.version == 1:
            preprocess_context.preprocess(s1, &(s1_proc[0].string))
            preprocess_context.preprocess(s2, &(s2_proc[0].string))
        else:
            s1 = processor(s1)
            s1_proc[0] = RF_StringWrapper(conv_sequence(s1), s1)
            s2 = processor(s2)
            s2_proc[0] = RF_StringWrapper(conv_sequence(s2), s2)

def distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates the minimum number of insertions, deletions, and substitutions
    required to change one sequence into the other according to Levenshtein with custom
    costs for insertion, deletion and substitution

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than max,
        max + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    Notes
    -----
    Depending on the input parameters different optimized implementation are used
    to improve the performance.

    Insertion = Deletion = Substitution:
      This is known as uniform Levenshtein distance and is the distance most commonly
      referred to as Levenshtein distance. The following implementation is used
      with a worst-case performance of ``O([N/64]M)``.

      - if max is 0 the similarity can be calculated using a direct comparision,
        since no difference between the strings is allowed.  The time complexity of
        this algorithm is ``O(N)``.

      - A common prefix/suffix of the two compared strings does not affect
        the Levenshtein distance, so the affix is removed before calculating the
        similarity.

      - If max is ≤ 3 the mbleven algorithm is used. This algorithm
        checks all possible edit operations that are possible under
        the threshold `max`. The time complexity of this algorithm is ``O(N)``.

      - If the length of the shorter string is ≤ 64 after removing the common affix
        Hyyrös' algorithm is used, which calculates the Levenshtein distance in
        parallel. The algorithm is described by [1]_. The time complexity of this
        algorithm is ``O(N)``.

      - If the length of the shorter string is ≥ 64 after removing the common affix
        a blockwise implementation of Myers' algorithm is used, which calculates
        the Levenshtein distance in parallel (64 characters at a time).
        The algorithm is described by [3]_. The time complexity of this
        algorithm is ``O([N/64]M)``.

    The following image shows a benchmark of the Levenshtein distance in multiple
    Python libraries. All of them are implemented either in C/C++ or Cython.
    The graph shows, that python-Levenshtein is the only library with a time
    complexity of ``O(NM)``, while all other libraries have a time complexity of
    ``O([N/64]M)``. Especially for long strings RapidFuzz is a lot faster than
    all the other tested libraries.

    .. image:: img/uniform_levenshtein.svg


    Insertion = Deletion, Substitution >= Insertion + Deletion:
      Since every Substitution can be performed as Insertion + Deletion, this variant
      of the Levenshtein distance only uses Insertions and Deletions. Therefore this
      variant is often referred to as InDel-Distance.  The following implementation
      is used with a worst-case performance of ``O([N/64]M)``.

      - if max is 0 the similarity can be calculated using a direct comparision,
        since no difference between the strings is allowed.  The time complexity of
        this algorithm is ``O(N)``.

      - if max is 1 and the two strings have a similar length, the similarity can be
        calculated using a direct comparision aswell, since a substitution would cause
        a edit distance higher than max. The time complexity of this algorithm
        is ``O(N)``.

      - A common prefix/suffix of the two compared strings does not affect
        the Levenshtein distance, so the affix is removed before calculating the
        similarity.

      - If max is ≤ 4 the mbleven algorithm is used. This algorithm
        checks all possible edit operations that are possible under
        the threshold `max`. As a difference to the normal Levenshtein distance this
        algorithm can even be used up to a threshold of 4 here, since the higher weight
        of substitutions decreases the amount of possible edit operations.
        The time complexity of this algorithm is ``O(N)``.

      - If the length of the shorter string is ≤ 64 after removing the common affix
        Hyyrös' lcs algorithm is used, which calculates the Indel distance in
        parallel. The algorithm is described by [4]_ and is extended with support
        for UTF32 in this implementation. The time complexity of this
        algorithm is ``O(N)``.

      - If the length of the shorter string is ≥ 64 after removing the common affix
        a blockwise implementation of the Hyyrös' lcs algorithm is used, which calculates
        the Levenshtein distance in parallel (64 characters at a time).
        The algorithm is described by [4]_. The time complexity of this
        algorithm is ``O([N/64]M)``.

    The following image shows a benchmark of the Indel distance in RapidFuzz
    and python-Levenshtein. Similar to the normal Levenshtein distance
    python-Levenshtein uses a implementation with a time complexity of ``O(NM)``,
    while RapidFuzz has a time complexity of ``O([N/64]M)``.

    .. image:: img/indel_levenshtein.svg


    Other weights:
      The implementation for other weights is based on Wagner-Fischer.
      It has a performance of ``O(N * M)`` and has a memory usage of ``O(N)``.
      Further details can be found in [2]_.

    References
    ----------
    .. [1] Hyyrö, Heikki. "A Bit-Vector Algorithm for Computing
           Levenshtein and Damerau Edit Distances."
           Nordic Journal of Computing, Volume 10 (2003): 29-39.
    .. [2] Wagner, Robert & Fischer, Michael
           "The String-to-String Correction Problem."
           J. ACM. 21. (1974): 168-173
    .. [3] Myers, Gene. "A fast bit-vector algorithm for approximate
           string matching based on dynamic programming."
           Journal of the ACM (JACM) 46.3 (1999): 395-415.
    .. [4] Hyyrö, Heikki. "Bit-Parallel LCS-length Computation Revisited"
           Proc. 15th Australasian Workshop on Combinatorial Algorithms (AWOCA 2004).

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> Levenshtein.distance("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.distance("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    cdef size_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef size_t c_score_cutoff = <size_t>-1 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)

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

cdef list levenshtein_opcodes_to_list(vector[LevenshteinOpcodes] ops):
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

    Examples
    --------
    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> for tag, src_pos, dest_pos in Levenshtein.Editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[1] s2[0]
    replace s1[3] s2[2]
     insert s1[6] s2[5]
    """

    cdef vector[LevenshteinEditOp] editops
    cdef size_t len_s1
    cdef size_t len_s2

    def __init__(self, s1, s2):
        """
        Create Editops

        Parameters
        ----------
        s1 : Sequence[Hashable]
            First string to compare.
        s2 : Sequence[Hashable]
            Second string to compare.
        """

        s1_proc = RF_StringWrapper(conv_sequence(s1))
        s2_proc = RF_StringWrapper(conv_sequence(s2))
        self.len_s1 = s1_proc.string.length
        self.len_s2 = s2_proc.string.length
        self.editops = levenshtein_editops_func(s1_proc.string, s2_proc.string)

    @classmethod
    def from_opcodes(cls, other):
        cdef Editops self = cls.__new__(cls)
        if not isinstance(other, Opcodes):
            raise TypeError("expected argument of type Opcodes")

        cdef Opcodes opcodes = other
        self.editops = opcodes_to_editops(opcodes.opcodes)
        self.len_s1 = opcodes.len_s1
        self.len_s2 = opcodes.len_s2
        return self

    def as_opcodes(self):
        cdef Opcodes opcodes = Opcodes.__new__(Opcodes)
        opcodes.opcodes = editops_to_opcodes(self.editops, self.len_s1, self.len_s2)
        opcodes.len_s1 = self.len_s1
        opcodes.len_s2 = self.len_s2
        return opcodes

    def as_list(self):
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

    cdef from_slice(self, slice subscript):
        cdef int start = subscript.start if subscript.start is not None else 0
        cdef int stop  = subscript.stop  if subscript.stop  is not None else self.editops.size()
        cdef int step  = subscript.step  if subscript.step  is not None else 1
        cdef Editops new_self = self.__class__.__new__(self.__class__)

        new_self.editops = vector_slice(self.editops, start, stop, step)
        return new_self

    def __getitem__(self, subscript):
        cdef int index
        if isinstance(subscript, slice):
            # disable this for now, since I am not sure how this should
            # affect conversion function
            raise TypeError("Editops does not support slicing")
            #return self.from_slice(subscript)
        else:
            index = subscript
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

    Examples
    --------
    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> for tag, i1, i2, j1, j2 in Levenshtein.Opcodes("qabxcd", "abycdf"):
    ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
    ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
     delete a[0:1] (q) b[0:0] ()
      equal a[1:3] (ab) b[0:2] (ab)
    replace a[3:4] (x) b[2:3] (y)
      equal a[4:6] (cd) b[3:5] (cd)
     insert a[6:6] () b[5:6] (f)
    """

    cdef vector[LevenshteinOpcodes] opcodes
    cdef size_t len_s1
    cdef size_t len_s2

    def __init__(self, s1, s2):
        """
        Create Opcodes

        Parameters
        ----------
        s1 : Sequence[Hashable]
            First string to compare.
        s2 : Sequence[Hashable]
            Second string to compare.
        """

        s1_proc = RF_StringWrapper(conv_sequence(s1))
        s2_proc = RF_StringWrapper(conv_sequence(s2))
        self.len_s1 = s1_proc.string.length
        self.len_s2 = s2_proc.string.length
        self.opcodes = editops_to_opcodes(
            levenshtein_editops_func(s1_proc.string, s2_proc.string), self.len_s1, self.len_s2
        )

    @classmethod
    def from_editops(cls, other):
        cdef Opcodes self = cls.__new__(cls)
        if not isinstance(other, Editops):
            raise TypeError("expected argument of type Editops")

        cdef Editops editops = other
        self.opcodes = editops_to_opcodes(editops.editops, editops.len_s1, editops.len_s2)
        self.len_s1 = editops.len_s1
        self.len_s2 = editops.len_s2
        return self

    def as_editops(self):
        cdef Editops editops = Editops.__new__(Editops)
        editops.editops = opcodes_to_editops(self.opcodes)
        editops.len_s1 = self.len_s1
        editops.len_s2 = self.len_s2
        return editops

    def as_list(self):
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

    cdef from_slice(self, slice subscript):
        cdef int start = subscript.start if subscript.start is not None else 0
        cdef int stop  = subscript.stop  if subscript.stop  is not None else self.editops.size()
        cdef int step  = subscript.step  if subscript.step  is not None else 1
        cdef Opcodes new_self = self.__class__.__new__(self.__class__)

        new_self.opcodes = vector_slice(self.opcodes, start, stop, step)
        return new_self

    def __getitem__(self, subscript):
        cdef int index
        if isinstance(subscript, slice):
            # disable this for now, since I am not sure how this should
            # affect conversion function
            raise TypeError("Opcodes does not support slicing")
            #return self.from_slice(subscript)
        else:
            index = subscript
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


def normalized_distance(s1, s2, *, weights=(1,1,1), processor=None, score_cutoff=None):
    """
    Calculates a normalized levenshtein distance using custom
    costs for insertion, deletion and substitution.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : Tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    See Also
    --------
    Levenshtein.distance : Levenshtein distance

    Notes
    -----
    The normalization of the Levenshtein distance is performed in the following way:

    .. math::
      :nowrap:

      \\begin{align*}
        dist_{max} &= \\begin{cases}
          min(len(s1), len(s2)) \cdot sub,       & \\text{if } sub \leq ins + del \\\\
          len(s1) \cdot del + len(s2) \cdot ins, & \\text{otherwise}
        \end{cases}\\\\[10pt]

        dist_{max} &= \\begin{cases}
          dist_{max} + (len(s1) - len(s2)) \cdot del, & \\text{if } len(s1) > len(s2) \\\\
          dist_{max} + (len(s2) - len(s1)) \cdot ins, & \\text{if } len(s1) < len(s2) \\\\
          dist_{max},                                 & \\text{if } len(s1) = len(s2)
        \end{cases}\\\\[10pt]

        ratio &= \\frac{distance(s1, s2)}{dist_{max}}
      \end{align*}

    Examples
    --------
    Find the normalized Levenshtein distance between two strings:

    >>> from rapidfuzz.algorithm.edit_based import Levenshtein
    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein")
    0.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein", score_cutoff=0.85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.normalized_distance("lewenstein", "levenshtein", weights=(1,1,2))
    0.85714285714285

     When a different processor is used s1 and s2 do not have to be strings

    >>> Levenshtein.normalized_distance(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    cdef RF_StringWrapper s1_proc, s2_proc
    if s1 is None or s2 is None:
        return 0

    cdef size_t insertion, deletion, substitution
    insertion = deletion = substitution = 1
    if weights is not None:
        insertion, deletion, substitution = weights

    cdef double c_score_cutoff = 0.0 if score_cutoff is None else score_cutoff

    preprocess_strings(s1, s2, processor, &s1_proc, &s2_proc)
    return normalized_levenshtein_func(s1_proc.string, s2_proc.string, insertion, deletion, substitution, c_score_cutoff)

cdef void KwargsDeinit(RF_Kwargs* self):
    free(<void*>dereference(self).context)

cdef bool LevenshteinKwargsInit(RF_Kwargs* self, dict kwargs) except False:
    cdef size_t insertion, deletion, substitution
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>malloc(sizeof(LevenshteinWeightTable))

    if not weights:
        raise MemoryError

    insertion, deletion, substitution = kwargs.get("weights", (1, 1, 1))
    dereference(weights).insert_cost = insertion
    dereference(weights).delete_cost = deletion
    dereference(weights).replace_cost = substitution
    dereference(self).context = weights
    dereference(self).dtor = KwargsDeinit
    return True

cdef bool GetScorerFlagsLevenshtein(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_U64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.u64 = 0
    dereference(scorer_flags).worst_score.u64 = SIZE_MAX
    return True

cdef bool GetScorerFlagsNormalizedLevenshtein(const RF_Kwargs* self, RF_ScorerFlags* scorer_flags) nogil except False:
    cdef LevenshteinWeightTable* weights = <LevenshteinWeightTable*>dereference(self).context
    dereference(scorer_flags).flags = RF_SCORER_FLAG_RESULT_F64
    if dereference(weights).insert_cost == dereference(weights).delete_cost:
        dereference(scorer_flags).flags |= RF_SCORER_FLAG_SYMMETRIC
    dereference(scorer_flags).optimal_score.f64 = 1.0
    dereference(scorer_flags).worst_score.f64 = 0
    return True

cdef RF_Scorer LevenshteinContext
LevenshteinContext.version = SCORER_STRUCT_VERSION
LevenshteinContext.kwargs_init = LevenshteinKwargsInit
LevenshteinContext.get_scorer_flags = GetScorerFlagsLevenshtein
LevenshteinContext.scorer_func_init = LevenshteinInit
distance._RF_Scorer = PyCapsule_New(&LevenshteinContext, NULL, NULL)

cdef RF_Scorer NormalizedLevenshteinContext
NormalizedLevenshteinContext.version = SCORER_STRUCT_VERSION
NormalizedLevenshteinContext.kwargs_init = LevenshteinKwargsInit
NormalizedLevenshteinContext.get_scorer_flags = GetScorerFlagsNormalizedLevenshtein
NormalizedLevenshteinContext.scorer_func_init = NormalizedLevenshteinInit
normalized_distance._RF_Scorer = PyCapsule_New(&NormalizedLevenshteinContext, NULL, NULL)

