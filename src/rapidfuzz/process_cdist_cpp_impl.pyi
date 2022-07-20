from typing import (
    Any,
    Callable,
    Hashable,
    Sequence,
    Iterable,
    Optional,
    Union,
    TypeVar
)
from rapidfuzz.fuzz import ratio

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")
_ResultType = Union[int, float]

FLOAT32: int
FLOAT64: int
INT8: int
INT16: int
INT32: int
INT64: int
UINT8: int
UINT16: int
UINT32: int
UINT64: int

try:
    import numpy as np

    def cdist(
        queries: Iterable[_S1],
        choices: Iterable[_S2],
        *,
        scorer: Callable[..., _ResultType] = ratio,
        processor: Optional[Callable[..., _StringType]] = None,
        score_cutoff: Optional[_ResultType] = None,
        dtype: Optional[np.dtype] = None,
        workers: int = 1,
        **kwargs: Any
    ) -> np.ndarray: ...

except ImportError:
    pass
