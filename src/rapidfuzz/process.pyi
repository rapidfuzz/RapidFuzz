

from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple, TypeVar, Union
from typing_extensions import Concatenate, ParamSpec

T = TypeVar("T")
C = TypeVar("C")
P = ParamSpec("P")

_ProcessorType = Optional[Union[bool, Callable[[Iterable[Hashable]], Iterable[Hashable]]]]

def extractOne(
    query: Iterable[Hashable],
    choices: Union[List[Iterable[Hashable]],
    Dict[Any, Iterable[Hashable]]],
    scorer: Optional[Callable[Concatenate[Iterable[Hashable], Iterable[Hashable], _ProcessorType, Optional[T], P], Union[float, int]]] = ...,
    processor: _ProcessorType = ...,
    score_cutoff: T = ...,
    *args: P.args,
    **kwargs: P.kwargs
) -> List[Tuple[str, Union[float, int], Any]]: ...
def extract(
    query: Iterable[Hashable],
    choices: Union[List[Iterable[Hashable]],
    Dict[Any, Iterable[Hashable]]],
    scorer: Optional[Callable[Concatenate[Iterable[Hashable], Iterable[Hashable], _ProcessorType, Optional[T], P], Union[float, int]]] = ...,
    processor: _ProcessorType = ...,
    limit: Optional[int] = ...,
    score_cutoff: T = ...,
    *args: P.args,
    **kwargs: P.kwargs
) -> List[Tuple[str, Union[float, int], Any]]: ...
def extract_iter(
    query: Iterable[Hashable],
    choices: Union[List[Iterable[Hashable]],
    Dict[Any, Iterable[Hashable]]],
    scorer: Optional[Callable[Concatenate[Iterable[Hashable], Iterable[Hashable], _ProcessorType, Optional[T], P], Union[float, int]]] = ...,
    processor: _ProcessorType = ...,
    limit: Optional[int] = ...,
    score_cutoff: T = ...,
    *args: P.args,
    **kwargs: P.kwargs
) -> List[Tuple[str, Union[float, int], Any]]: ...

