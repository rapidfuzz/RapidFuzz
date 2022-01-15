#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from rapidfuzz.distance import Levenshtein, Editops, Opcodes

def test_editops_comparision():
    """
    test comparision with Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops == ops
    assert not (ops != ops)
    assert ops == ops[:]
    assert not (ops != ops[:])

def test_editops_get_index():
    """
    test __getitem__ with index of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    ops_list = [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]

    assert ops[0] == ops_list[0]
    assert ops[1] == ops_list[1]
    assert ops[2] == ops_list[2]
    assert ops[3] == ops_list[3]
    assert ops[4] == ops_list[4]

    assert ops[-1] == ops_list[-1]
    assert ops[-2] == ops_list[-2]
    assert ops[-3] == ops_list[-3]
    assert ops[-4] == ops_list[-4]
    assert ops[-5] == ops_list[-5]

    with pytest.raises(IndexError):
        ops[5]
    with pytest.raises(IndexError):
        ops[-6]

def test_editops_slicing():
    """
    test __getitem__ with slices of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    ops_list = [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]

    assert ops[:]       == Editops(ops_list, 7, 9)
    assert ops[1:]      == Editops(ops_list[1:], 7, 9)
    assert ops[:3]      == Editops(ops_list[:3], 7, 9)
    assert ops[2:4]     == Editops(ops_list[2:4], 7, 9)
    assert ops[2:3]     == Editops(ops_list[2:3], 7, 9)
    assert ops[2:2]     == Editops(ops_list[2:2], 7, 9)
    assert ops[3:-1]    == Editops(ops_list[3:-1], 7, 9)
    assert ops[-2:-1]   == Editops(ops_list[-2:-1], 7, 9)
    assert ops[-2:-2]   == Editops(ops_list[-2:-2], 7, 9)
    assert ops[-2:-3]   == Editops(ops_list[-2:-3], 7, 9)
    assert ops[1:4:2]   == Editops(ops_list[1:4:2], 7, 9)
    assert ops[1:3:2]   == Editops(ops_list[1:3:2], 7, 9)
    assert ops[1:4:-1]  == Editops(ops_list[1:4:-1], 7, 9)
    assert ops[5:-7:-1] == Editops(ops_list[5:-7:-1], 7, 9)
    assert ops[5:-7:-1] == ops.reverse()
    assert ops[4:-4:-1] == Editops(ops_list[4:-4:-1], 7, 9)
    assert ops[3:-5:-1] == Editops(ops_list[3:-5:-1], 7, 9)
    assert ops[3:-5:-2] == Editops(ops_list[3:-5:-2], 7, 9)
    with pytest.raises(ValueError):
        ops[::0]

def test_editops_setitem():
    """
    test __setitem__ with Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    ops_list = [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    assert ops == Editops(ops_list, 7, 9)
    ops[1] = ops_list[1] = ('insert', 6, 5)
    assert ops == Editops(ops_list, 7, 9)

def test_editops_inversion():
    """
    test correct inversion of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops.as_list() == [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    assert ops.inverse().as_list() == [('insert', 1, 1), ('replace', 1, 2),
        ('delete', 5, 6), ('delete', 6, 6), ('delete', 7, 6)]

def test_opcodes_comparision():
    """
    test comparision with Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops == ops
    assert not (ops != ops)
    assert ops == ops[:]
    assert not (ops != ops[:])

def test_opcode_get_index():
    """
    test __getitem__ with index of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    ops_list = [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)]

    assert ops[0] == ops_list[0]
    assert ops[1] == ops_list[1]
    assert ops[2] == ops_list[2]
    assert ops[3] == ops_list[3]
    assert ops[4] == ops_list[4]
    assert ops[5] == ops_list[5]

    assert ops[-1] == ops_list[-1]
    assert ops[-2] == ops_list[-2]
    assert ops[-3] == ops_list[-3]
    assert ops[-4] == ops_list[-4]
    assert ops[-5] == ops_list[-5]
    assert ops[-6] == ops_list[-6]

    with pytest.raises(IndexError):
        ops[6]
    with pytest.raises(IndexError):
        ops[-7]

def test_opcodes_slicing():
    """
    test __getitem__ with slices of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    ops_list = [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)]
    assert ops[:]       == Opcodes(ops_list, 7, 9)
    assert ops[1:]      == Opcodes(ops_list[1:], 7, 9)
    assert ops[:3]      == Opcodes(ops_list[:3], 7, 9)
    assert ops[2:4]     == Opcodes(ops_list[2:4], 7, 9)
    assert ops[2:3]     == Opcodes(ops_list[2:3], 7, 9)
    assert ops[2:2]     == Opcodes(ops_list[2:2], 7, 9)
    assert ops[3:-1]    == Opcodes(ops_list[3:-1], 7, 9)
    assert ops[-2:-1]   == Opcodes(ops_list[-2:-1], 7, 9)
    assert ops[-2:-2]   == Opcodes(ops_list[-2:-2], 7, 9)
    assert ops[-2:-3]   == Opcodes(ops_list[-2:-3], 7, 9)
    assert ops[1:4:2]   == Opcodes(ops_list[1:4:2], 7, 9)
    assert ops[1:3:2]   == Opcodes(ops_list[1:3:2], 7, 9)
    assert ops[1:4:-1]  == Opcodes(ops_list[1:4:-1], 7, 9)
    assert ops[5:-7:-1] == Opcodes(ops_list[5:-7:-1], 7, 9)
    assert ops[5:-7:-1] == ops.reverse()
    assert ops[4:-4:-1] == Opcodes(ops_list[4:-4:-1], 7, 9)
    assert ops[3:-5:-1] == Opcodes(ops_list[3:-5:-1], 7, 9)
    assert ops[3:-5:-2] == Opcodes(ops_list[3:-5:-2], 7, 9)
    with pytest.raises(ValueError):
        ops[::0]

def test_opcodes_setitem():
    """
    test __setitem__ with Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    ops_list = [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)]
    assert ops == Opcodes(ops_list, 7, 9)
    ops[1] = ops_list[1] = ('replace', 2, 3, 1, 2)
    assert ops == Opcodes(ops_list, 7, 9)

def test_opcode_inversion():
    """
    test correct inversion of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops == Opcodes([('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)], 7, 9)

    assert ops.inverse().as_list() == [('equal', 0, 1, 0, 1), ('insert', 1, 1, 1, 2),
        ('replace', 1, 2, 2, 3), ('equal', 2, 5, 3, 6), ('delete', 5, 8, 6, 6), ('equal', 8, 9, 6, 7)]


if __name__ == '__main__':
    unittest.main()