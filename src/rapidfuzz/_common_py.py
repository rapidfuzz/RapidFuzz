# SPDX-License-Identifier: MIT
# Copyright (C) 2023 Max Bachmann

from __future__ import annotations

from array import array
from typing import Hashable, Sequence


def conv_sequence(s: Sequence[Hashable]) -> Sequence[Hashable]:
    if isinstance(s, str):
        return [ord(x) for x in s]

    if isinstance(s, bytes):
        return s

    if isinstance(s, array):
        if s.typecode == "u":
            return [ord(x) for x in s]

        return s

    if s is None:
        return s

    res = []
    for elem in s:
        if isinstance(elem, str) and len(elem) == 1:
            res.append(ord(elem))
        elif isinstance(elem, int) and elem == -1:
            res.append(-1)
        else:
            res.append(hash(elem))

    return res


def conv_sequences(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> Sequence[Hashable]:
    if isinstance(s1, str) and isinstance(s2, str):
        return s1, s2

    if isinstance(s1, bytes) and isinstance(s2, bytes):
        return s1, s2

    return conv_sequence(s1), conv_sequence(s2)
