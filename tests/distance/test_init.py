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
    assert ops == ops.copy()
    assert not (ops != ops.copy())

def test_editops_get_index():
    """
    test __getitem__ with index of Editops
    """
    ops = Editops([('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)], 7, 9)

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

def test_editops_inversion():
    """
    test correct inversion of Editops
    """
    ops = Editops([('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)], 7, 9)

    assert ops.inverse().as_list() == [('insert', 1, 1), ('replace', 1, 2),
        ('delete', 5, 6), ('delete', 6, 6), ('delete', 7, 6)]

def test_opcodes_comparision():
    """
    test comparision with Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops == ops
    assert not (ops != ops)
    assert ops == ops.copy()
    assert not (ops != ops.copy())

def test_opcode_get_index():
    """
    test __getitem__ with index of Opcodes
    """
    ops = Opcodes([('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)], 7, 9)

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

def test_opcode_inversion():
    """
    test correct inversion of Opcodes
    """
    ops = Opcodes([('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8), ('equal', 6, 7, 8, 9)], 7, 9)

    assert ops.inverse().as_list() == [('equal', 0, 1, 0, 1), ('insert', 1, 1, 1, 2),
        ('replace', 1, 2, 2, 3), ('equal', 2, 5, 3, 6), ('delete', 5, 8, 6, 6), ('equal', 8, 9, 6, 7)]

def test_list_initialization():
    """
    test whether list initialization works correctly
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    ops2 = Opcodes(ops.as_list(), ops.src_len, ops.dest_len)
    assert ops == ops2

    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    ops2 = Editops(ops.as_list(), ops.src_len, ops.dest_len)
    assert ops == ops2

    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    ops2 = Editops(ops.as_list(), ops.src_len, ops.dest_len)
    assert ops.as_editops() == ops2

    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    ops2 = Opcodes(ops.as_list(), ops.src_len, ops.dest_len)
    assert ops.as_opcodes() == ops2

    ops = Levenshtein.editops("skdsakldsakdlasda", "djkajkdfkdgkhdfjrmecsidjf")
    ops2 = Opcodes(ops.as_list(), ops.src_len, ops.dest_len)
    assert ops.as_opcodes() == ops2

if __name__ == '__main__':
    unittest.main()