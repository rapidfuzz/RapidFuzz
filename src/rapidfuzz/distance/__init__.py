# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.utils import _fallback_import

_mod = "rapidfuzz.distance._initialize"
Editop = _fallback_import(_mod, "Editop")
Editops = _fallback_import(_mod, "Editops")
Opcode = _fallback_import(_mod, "Opcode")
Opcodes = _fallback_import(_mod, "Opcodes")
ScoreAlignment = _fallback_import(_mod, "ScoreAlignment")
MatchingBlock = _fallback_import(_mod, "MatchingBlock")

from . import Hamming, Indel, Jaro, JaroWinkler, Levenshtein, LCSseq, DamerauLevenshtein
