from __future__ import annotations

from rapidfuzz._utils import fallback_import as _fallback_import

_mod = "rapidfuzz.distance._initialize"
Editop = _fallback_import(_mod, "Editop", set_attrs=False)
Editops = _fallback_import(_mod, "Editops", set_attrs=False)
Opcode = _fallback_import(_mod, "Opcode", set_attrs=False)
Opcodes = _fallback_import(_mod, "Opcodes", set_attrs=False)
ScoreAlignment = _fallback_import(_mod, "ScoreAlignment", set_attrs=False)
MatchingBlock = _fallback_import(_mod, "MatchingBlock", set_attrs=False)

__all__ = [
    "Editop",
    "Editops",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "MatchingBlock",
]
