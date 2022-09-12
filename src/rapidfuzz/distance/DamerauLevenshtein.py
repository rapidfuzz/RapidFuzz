# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from rapidfuzz._utils import (
    fallback_import as _fallback_import,
    default_distance_attribute as _dist_attr,
    default_similarity_attribute as _sim_attr,
    default_normalized_distance_attribute as _norm_dist_attr,
    default_normalized_similarity_attribute as _norm_sim_attr,
)

_mod = "rapidfuzz.distance.DamerauLevenshtein"
distance = _fallback_import(_mod, "distance")
similarity = _fallback_import(_mod, "similarity")
normalized_distance = _fallback_import(_mod, "normalized_distance")
normalized_similarity = _fallback_import(_mod, "normalized_similarity")

distance._RF_ScorerPy = _dist_attr
similarity._RF_ScorerPy = _sim_attr
normalized_distance._RF_ScorerPy = _norm_dist_attr
normalized_similarity._RF_ScorerPy = _norm_sim_attr
