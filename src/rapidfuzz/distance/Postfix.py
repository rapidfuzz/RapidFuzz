# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

from rapidfuzz._utils import default_distance_attribute as _dist_attr
from rapidfuzz._utils import default_normalized_distance_attribute as _norm_dist_attr
from rapidfuzz._utils import default_normalized_similarity_attribute as _norm_sim_attr
from rapidfuzz._utils import default_similarity_attribute as _sim_attr
from rapidfuzz._utils import fallback_import as _fallback_import

_mod = "rapidfuzz.distance.Postfix"
distance = _fallback_import(_mod, "distance", cached_scorer_call=_dist_attr)
similarity = _fallback_import(_mod, "similarity", cached_scorer_call=_sim_attr)
normalized_distance = _fallback_import(
    _mod, "normalized_distance", cached_scorer_call=_norm_dist_attr
)
normalized_similarity = _fallback_import(
    _mod, "normalized_similarity", cached_scorer_call=_norm_sim_attr
)
