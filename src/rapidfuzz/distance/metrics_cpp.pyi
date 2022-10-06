from __future__ import annotations

from typing import Callable, Hashable, Optional, Sequence, Tuple, TypeVar

from rapidfuzz.distance import Editops, Opcodes

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

def levenshtein_distance(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def levenshtein_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def levenshtein_similarity(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def levenshtein_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    weights: Optional[Tuple[int, int, int]] = (1, 1, 1),
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def levenshtein_editops(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_hint: Optional[int] = None,
) -> Editops: ...
def levenshtein_opcodes(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_hint: Optional[int] = None,
) -> Opcodes: ...
def indel_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def indel_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def indel_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def indel_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def indel_editops(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Editops: ...
def indel_opcodes(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Opcodes: ...
def lcs_seq_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def lcs_seq_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def lcs_seq_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def lcs_seq_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def lcs_seq_editops(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Editops: ...
def lcs_seq_opcodes(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Opcodes: ...
def hamming_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def hamming_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def hamming_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def hamming_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def hamming_editops(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Editops: ...
def hamming_opcodes(
    s1: _S1, s2: _S2, *, processor: Optional[Callable[..., _StringType]] = None
) -> Opcodes: ...
def damerau_levenshtein_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def damerau_levenshtein_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def damerau_levenshtein_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def damerau_levenshtein_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def osa_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def osa_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def osa_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> int: ...
def osa_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def jaro_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> float: ...
def jaro_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def jaro_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> float: ...
def jaro_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def jaro_winkler_distance(
    s1: _S1,
    s2: _S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> float: ...
def jaro_winkler_normalized_distance(
    s1: _S1,
    s2: _S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
def jaro_winkler_similarity(
    s1: _S1,
    s2: _S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[int] = None,
) -> float: ...
def jaro_winkler_normalized_similarity(
    s1: _S1,
    s2: _S2,
    *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[..., _StringType]] = None,
    score_cutoff: Optional[float] = 0,
) -> float: ...
