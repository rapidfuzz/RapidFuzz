from typing import Callable, Hashable, Sequence, Optional, Union, TypeVar, Tuple, List
from rapidfuzz.algorithm.edit_based import Editops, Opcodes

_StringType = Sequence[Hashable]
S1 = TypeVar("S1")
S2 = TypeVar("S2")

def distance(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]] = None,
    max: Optional[int] = None) -> int: ...

def normalized_distance(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def editops(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]] = None) -> Editops: ...

def opcodes(
    s1: S1, s2: S2, *,
    processor: Optional[Callable[[Union[S1, S2]], _StringType]] = None) -> Opcodes: ...
