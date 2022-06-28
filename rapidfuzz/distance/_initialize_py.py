# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann


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
