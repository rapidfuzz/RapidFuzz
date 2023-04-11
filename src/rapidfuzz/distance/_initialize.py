from __future__ import annotations

from rapidfuzz._utils import fallback_import as _fallback_import

_mod = "rapidfuzz.distance._initialize"
Editop = _fallback_import(_mod, "Editop")
Editops = _fallback_import(_mod, "Editops")
Opcode = _fallback_import(_mod, "Opcode")
Opcodes = _fallback_import(_mod, "Opcodes")
ScoreAlignment = _fallback_import(_mod, "ScoreAlignment")
MatchingBlock = _fallback_import(_mod, "MatchingBlock")

__all__ = [
    "Editop",
    "Editops",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "MatchingBlock",
]
