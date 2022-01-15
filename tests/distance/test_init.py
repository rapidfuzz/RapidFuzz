#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from rapidfuzz.distance import Levenshtein

def test_editops_get_index():
    """
    test __getitem__ with index of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops[0] == ('delete', 1, 1)
    assert ops[1] == ('replace', 2, 1)
    assert ops[2] == ('insert', 6, 5)
    assert ops[3] == ('insert', 6, 6)
    assert ops[4] == ('insert', 6, 7)

    assert ops[-1]  == ('insert', 6, 7)
    assert ops[-2] == ('insert', 6, 6)
    assert ops[-3] == ('insert', 6, 5)
    assert ops[-4] == ('replace', 2, 1)
    assert ops[-5] == ('delete', 1, 1)

    with pytest.raises(IndexError):
        ops[5]
    with pytest.raises(IndexError):
        ops[-6]

def test_editops_slicing():
    """
    test __getitem__ with slices of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops[:].as_list() == [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    assert ops[1:].as_list() == [('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    assert ops[:3].as_list() == [('delete', 1, 1), ('replace', 2, 1), ('insert', 6, 5)]
    assert ops[2:4].as_list() == [('insert', 6, 5), ('insert', 6, 6)]
    assert ops[2:3].as_list() == [('insert', 6, 5)]
    assert ops[2:2].as_list() == []
    assert ops[3:-1].as_list() == [('insert', 6, 6)]
    assert ops[-2:-1].as_list() == [('insert', 6, 6)]
    assert ops[-2:-2].as_list() == []
    assert ops[-2:-3].as_list() == []
    assert ops[1:4:2].as_list() == [('replace', 2, 1), ('insert', 6, 6)]
    assert ops[1:3:2].as_list() == [('replace', 2, 1)]
    assert ops[1:4:-1].as_list() == []
    assert ops[4:-6:-1].as_list() == [('insert', 6, 7), ('insert', 6, 6),
        ('insert', 6, 5), ('replace', 2, 1), ('delete', 1, 1)]
    assert ops[4:-6:-1] == ops.reverse()
    assert ops[4:-4:-1].as_list() == [('insert', 6, 7), ('insert', 6, 6), ('insert', 6, 5)]
    assert ops[3:-5:-1].as_list() == [('insert', 6, 6), ('insert', 6, 5), ('replace', 2, 1)]
    assert ops[3:-5:-2].as_list() == [('insert', 6, 6), ('replace', 2, 1)]
    with pytest.raises(ValueError):
        ops[::0]

def test_editops_setitem():
    """
    test __setitem__ with Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops.as_list() == [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    ops[1] = ('insert', 6, 5)
    assert ops.as_list() == [('delete', 1, 1), ('insert', 6, 5),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]

def test_editops_inversion():
    """
    test correct inversion of Editops
    """
    ops = Levenshtein.editops("aaabaaa", "abbaaabba")
    assert ops.as_list() == [('delete', 1, 1), ('replace', 2, 1),
        ('insert', 6, 5), ('insert', 6, 6), ('insert', 6, 7)]
    assert ops.inverse().as_list() == [('insert', 1, 1), ('replace', 1, 2),
        ('delete', 5, 6), ('delete', 6, 6), ('delete', 7, 6)]


def test_opcode_get_index():
    """
    test __getitem__ with index of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops[0] == ('equal', 0, 1, 0, 1)
    assert ops[1] == ('delete', 1, 2, 1, 1)
    assert ops[2] == ('replace', 2, 3, 1, 2)
    assert ops[3] == ('equal', 3, 6, 2, 5)
    assert ops[4] == ('insert', 6, 6, 5, 8)

    assert ops[-1] == ('insert', 6, 6, 5, 8)
    assert ops[-2] == ('equal', 3, 6, 2, 5)
    assert ops[-3] == ('replace', 2, 3, 1, 2)
    assert ops[-4] == ('delete', 1, 2, 1, 1)
    assert ops[-5] == ('equal', 0, 1, 0, 1)

    with pytest.raises(IndexError):
        ops[5]
    with pytest.raises(IndexError):
        ops[-6]

def test_opcodes_slicing():
    """
    test __getitem__ with slices of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops[:].as_list() == [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8)]
    assert ops[1:].as_list() == [('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8)]
    assert ops[:3].as_list() == [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2)]
    assert ops[2:4].as_list() == [('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5)]
    assert ops[2:3].as_list() == [('replace', 2, 3, 1, 2)]
    assert ops[2:2].as_list() == []
    assert ops[3:-1].as_list() == [('equal', 3, 6, 2, 5)]
    assert ops[-2:-1].as_list() == [('equal', 3, 6, 2, 5)]
    assert ops[-2:-2].as_list() == []
    assert ops[-2:-3].as_list() == []
    assert ops[1:4:2].as_list() == [('delete', 1, 2, 1, 1), ('equal', 3, 6, 2, 5)]
    assert ops[1:3:2].as_list() == [('delete', 1, 2, 1, 1)]
    assert ops[1:4:-1].as_list() == []
    assert ops[4:-6:-1].as_list() == [('insert', 6, 6, 5, 8), ('equal', 3, 6, 2, 5),
        ('replace', 2, 3, 1, 2), ('delete', 1, 2, 1, 1), ('equal', 0, 1, 0, 1)]
    assert ops[4:-6:-1] == ops.reverse()
    assert ops[4:-4:-1].as_list() == [('insert', 6, 6, 5, 8), ('equal', 3, 6, 2, 5),
        ('replace', 2, 3, 1, 2)]
    assert ops[3:-5:-1].as_list() == [('equal', 3, 6, 2, 5), ('replace', 2, 3, 1, 2),
        ('delete', 1, 2, 1, 1)]
    assert ops[3:-5:-2].as_list() == [('equal', 3, 6, 2, 5), ('delete', 1, 2, 1, 1)]
    with pytest.raises(ValueError):
        ops[::0]

def test_opcodes_setitem():
    """
    test __setitem__ with Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops.as_list() == [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8)]
    ops[1] = ('replace', 2, 3, 1, 2)
    assert ops.as_list() == [('equal', 0, 1, 0, 1), ('replace', 2, 3, 1, 2),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8)]

def test_opcode_inversion():
    """
    test correct inversion of Opcodes
    """
    ops = Levenshtein.opcodes("aaabaaa", "abbaaabba")
    assert ops.as_list() == [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1),
        ('replace', 2, 3, 1, 2), ('equal', 3, 6, 2, 5), ('insert', 6, 6, 5, 8)]

    assert ops.inverse().as_list() == [('equal', 0, 1, 0, 1), ('insert', 1, 1, 1, 2),
        ('replace', 1, 2, 2, 3), ('equal', 2, 5, 3, 6), ('delete', 5, 8, 6, 6)]


if __name__ == '__main__':
    unittest.main()