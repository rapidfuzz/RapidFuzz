# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz._utils import fallback_import as _fallback_import

_mod = "rapidfuzz.distance._initialize"
Editop = _fallback_import(_mod, "Editop", False)
Editops = _fallback_import(_mod, "Editops", False)
Opcode = _fallback_import(_mod, "Opcode", False)
Opcodes = _fallback_import(_mod, "Opcodes", False)
ScoreAlignment = _fallback_import(_mod, "ScoreAlignment", False)
MatchingBlock = _fallback_import(_mod, "MatchingBlock", False)

from . import Hamming, Indel, Jaro, JaroWinkler, Levenshtein, LCSseq, DamerauLevenshtein
