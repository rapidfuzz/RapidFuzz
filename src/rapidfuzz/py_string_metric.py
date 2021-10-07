

import sys
from rapidfuzz.details.common import (
        popcount,
        norm_distance,
        remove_common_affix
)

if sys.version_info >= (3,0,0):
    def indel_distance(s1, s2):
        block = {ch: 0 for ch in s2}
        for i, ch in enumerate(s2):
            block[ch] |= 1 << i

        S = ~0

        for ch in s1:
            Matches = block.get(ch, 0)
            u = S & Matches
            S = (S + u) | (S - u)

        return len(s1) + len(s2) - 2 * popcount(~S)
else:
    from collections import defaultdict
    def indel_distance(s1, s2):
        zero = long(0)
        one = long(1)
        block = defaultdict(long)
        for i, ch in enumerate(s2):
            block[ch] |= one << i

        S = ~zero

        for ch in s1:
            Matches = block.get(ch, zero)
            u = S & Matches
            S = (S + u) | (S - u)
    
        return len(s1) + len(s2) - 2 * popcount(~S)

def normalized_indel_distance(s1, s2, score_cutoff):
    lensum = len(s1) + len(s2)
    s1, s2 = remove_common_affix(s1, s2)

    if not s1 and not s2:
        return 100.0
    if not s1:
        return norm_distance(len(s2), lensum, score_cutoff)
    if not s2:
        return norm_distance(len(s1), lensum, score_cutoff)

    dist = indel_distance(s1, s2)
    return norm_distance(dist, lensum, score_cutoff)


