# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz.utils import _fallback_import

_mod = "rapidfuzz.distance.DamerauLevenshtein"
distance = _fallback_import(_mod, "distance")
similarity = _fallback_import(_mod, "similarity")
normalized_distance = _fallback_import(_mod, "normalized_distance")
normalized_similarity = _fallback_import(_mod, "normalized_similarity")
