# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz._utils import fallback_import as _fallback_import

_mod = "rapidfuzz.distance.metrics"
distance = _fallback_import(_mod, "indel_distance")
similarity = _fallback_import(_mod, "indel_similarity")
normalized_distance = _fallback_import(_mod, "indel_normalized_distance")
normalized_similarity = _fallback_import(_mod, "indel_normalized_similarity")
editops = _fallback_import(_mod, "indel_editops")
opcodes = _fallback_import(_mod, "indel_opcodes")
