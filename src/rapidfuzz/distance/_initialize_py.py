# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann


class MatchingBlock:
    def __init__(self, a, b, size):
        self.a = a
        self.b = b
        self.size = size

    def __len__(self):
        return 3

    def __eq__(self, other):
        if len(other) != 3:
            return False

        return other[0] == self.a and other[1] == self.b and other[2] == self.size

    def __getitem__(self, i):
        if i == 0 or i == -3:
            return self.a
        if i == 1 or i == -2:
            return self.b
        if i == 2 or i == -1:
            return self.size

        raise IndexError("MatchingBlock index out of range")

    def __repr__(self):
        return f"MatchingBlock(a={self.a}, b={self.b}, size={self.size})"


class Editop:
    """
    Tuple like object describing an edit operation.
    It is in the form (tag, src_pos, dest_pos)

    The tags are strings, with these meanings:

    +-----------+---------------------------------------------------+
    | tag       | explanation                                       |
    +===========+===================================================+
    | 'replace' | src[src_pos] should be replaced by dest[dest_pos] |
    +-----------+---------------------------------------------------+
    | 'delete'  | src[src_pos] should be deleted                    |
    +-----------+---------------------------------------------------+
    | 'insert'  | dest[dest_pos] should be inserted at src[src_pos] |
    +-----------+---------------------------------------------------+
    """

    def __init__(self, tag, src_pos, dest_pos):
        self.tag = tag
        self.src_pos = src_pos
        self.dest_pos = dest_pos

    def __len__(self):
        return 3

    def __eq__(self, other):
        if len(other) != 3:
            return False

        return (
            other[0] == self.tag
            and other[1] == self.src_pos
            and other[2] == self.dest_pos
        )

    def __getitem__(self, i):
        if i == 0 or i == -3:
            return self.tag
        if i == 1 or i == -2:
            return self.src_pos
        if i == 2 or i == -1:
            return self.dest_pos

        raise IndexError("Editop index out of range")

    def __repr__(self):
        return (
            f"Editop(tag={self.tag}, src_pos={self.src_pos}, dest_pos={self.dest_pos})"
        )


class Editops:
    """
    List like object of Editos describing how to turn s1 into s2.
    """

    def __init__(self, editops=None, src_len=0, dest_len=0):
        raise NotImplementedError

    @classmethod
    def from_opcodes(cls, opcodes):
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
        raise NotImplementedError

    def as_opcodes(self):
        """
        Convert to Opcodes

        Returns
        -------
        opcodes : Opcodes
            Editops converted to Opcodes
        """
        raise NotImplementedError

    def as_matching_blocks(self):
        raise NotImplementedError

    def as_list(self):
        """
        Convert Editops to a list of tuples.

        This is the equivalent of ``[x for x in editops]``
        """
        raise NotImplementedError

    def copy(self):
        """
        performs copy of Editops
        """
        raise NotImplementedError

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
        [Editop(tag=delete, src_pos=0, dest_pos=0),
         Editop(tag=replace, src_pos=3, dest_pos=2),
         Editop(tag=insert, src_pos=4, dest_pos=3)]

        >>> Levenshtein.editops('spam', 'park').inverse()
        [Editop(tag=insert, src_pos=0, dest_pos=0),
         Editop(tag=replace, src_pos=2, dest_pos=3),
         Editop(tag=delete, src_pos=3, dest_pos=4)]
        """
        raise NotImplementedError

    def remove_subsequence(self, subsequence):
        raise NotImplementedError

    def apply(self, source_string, destination_string):
        raise NotImplementedError

    @property
    def src_len(self):
        raise NotImplementedError

    @src_len.setter
    def src_len(self, value):
        raise NotImplementedError

    @property
    def dest_len(self):
        raise NotImplementedError

    @dest_len.setter
    def dest_len(self, value):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __delitem__(self, item) -> None:
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __repr__(self):
        return "[" + ", ".join(repr(op) for op in self) + "]"


class Opcode:
    """
    Tuple like object describing an edit operation.
    It is in the form (tag, src_start, src_end, dest_start, dest_end)

    The tags are strings, with these meanings:

    +-----------+-----------------------------------------------------+
    | tag       | explanation                                         |
    +===========+=====================================================+
    | 'replace' | src[src_start:src_end] should be                    |
    |           | replaced by dest[dest_start:dest_end]               |
    +-----------+-----------------------------------------------------+
    | 'delete'  | src[src_start:src_end] should be deleted.           |
    |           | Note that dest_start==dest_end in this case.        |
    +-----------+-----------------------------------------------------+
    | 'insert'  | dest[dest_start:dest_end] should be inserted        |
    |           | at src[src_start:src_start].                        |
    |           | Note that src_start==src_end in this case.          |
    +-----------+-----------------------------------------------------+
    | 'equal'   | src[src_start:src_end] == dest[dest_start:dest_end] |
    +-----------+-----------------------------------------------------+

    Note
    ----
    Opcode is compatible with the tuples returned by difflib's SequenceMatcher to make them
    interoperable
    """

    def __init__(self, tag, src_start, src_end, dest_start, dest_end):
        self.tag = tag
        self.src_start = src_start
        self.src_end = src_end
        self.dest_start = dest_start
        self.dest_end = dest_end

    def __len__(self):
        return 5

    def __eq__(self, other):
        if len(other) != 5:
            return False

        return (
            other[0] == self.tag
            and other[1] == self.src_start
            and other[2] == self.src_end
            and other[3] == self.dest_start
            and other[4] == self.dest_end
        )

    def __getitem__(self, i):
        if i == 0 or i == -5:
            return self.tag
        if i == 1 or i == -4:
            return self.src_start
        if i == 2 or i == -3:
            return self.src_end
        if i == 3 or i == -2:
            return self.dest_start
        if i == 4 or i == -1:
            return self.dest_end

        raise IndexError("Opcode index out of range")

    def __repr__(self):
        return f"Opcode(tag={self.tag}, src_start={self.src_start}, src_end={self.src_end}, dest_start={self.dest_start}, dest_end={self.dest_end})"


class Opcodes:
    """
    List like object of Opcodes describing how to turn s1 into s2.
    The first Opcode has src_start == dest_start == 0, and remaining tuples
    have src_start == the src_end from the tuple preceding it,
    and likewise for dest_start == the previous dest_end.
    """

    def __init__(self, opcodes=None, src_len=0, dest_len=0):
        raise NotImplementedError

    @classmethod
    def from_editops(cls, editops):
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
        raise NotImplementedError

    def as_editops(self):
        """
        Convert Opcodes to Editops

        Returns
        -------
        editops : Editops
            Opcodes converted to Editops
        """
        raise NotImplementedError

    def as_matching_blocks(self):
        raise NotImplementedError

    def as_list(self):
        """
        Convert Opcodes to a list of tuples, which is compatible
        with the opcodes of difflibs SequenceMatcher.

        This is the equivalent of ``[x for x in opcodes]``
        """
        raise NotImplementedError

    def copy(self):
        """
        performs copy of Opcodes
        """
        raise NotImplementedError

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
        [Opcode(tag=delete, src_start=0, src_end=1, dest_start=0, dest_end=0),
         Opcode(tag=equal, src_start=1, src_end=3, dest_start=0, dest_end=2),
         Opcode(tag=replace, src_start=3, src_end=4, dest_start=2, dest_end=3),
         Opcode(tag=insert, src_start=4, src_end=4, dest_start=3, dest_end=4)]

        >>> Levenshtein.opcodes('spam', 'park').inverse()
        [Opcode(tag=insert, src_start=0, src_end=0, dest_start=0, dest_end=1),
         Opcode(tag=equal, src_start=0, src_end=2, dest_start=1, dest_end=3),
         Opcode(tag=replace, src_start=2, src_end=3, dest_start=3, dest_end=4),
         Opcode(tag=delete, src_start=3, src_end=4, dest_start=4, dest_end=4)]
        """
        raise NotImplementedError

    def apply(self, source_string, destination_string):
        raise NotImplementedError

    @property
    def src_len(self):
        raise NotImplementedError

    @src_len.setter
    def src_len(self, value):
        raise NotImplementedError

    @property
    def dest_len(self):
        raise NotImplementedError

    @dest_len.setter
    def dest_len(self, value):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __repr__(self):
        return "[" + ", ".join(repr(op) for op in self) + "]"


class ScoreAlignment:
    """
    Tuple like object describing the position of the compared strings in
    src and dest.

    It indicates that the score has been calculated between
    src[src_start:src_end] and dest[dest_start:dest_end]
    """

    def __init__(self, score, src_start, src_end, dest_start, dest_end):
        self.score = score
        self.src_start = src_start
        self.src_end = src_end
        self.dest_start = dest_start
        self.dest_end = dest_end

    def __len__(self):
        return 5

    def __eq__(self, other):
        if len(other) != 5:
            return False

        return (
            other[0] == self.score
            and other[1] == self.src_start
            and other[2] == self.src_end
            and other[3] == self.dest_start
            and other[4] == self.dest_end
        )

    def __getitem__(self, i):
        if i == 0 or i == -5:
            return self.score
        if i == 1 or i == -4:
            return self.src_start
        if i == 2 or i == -3:
            return self.src_end
        if i == 3 or i == -2:
            return self.dest_start
        if i == 4 or i == -1:
            return self.dest_end

        raise IndexError("Opcode index out of range")

    def __repr__(self):
        return f"ScoreAlignment(score={self.score}, src_start={self.src_start}, src_end={self.src_end}, dest_start={self.dest_start}, dest_end={self.dest_end})"
