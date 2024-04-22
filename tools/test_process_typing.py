from __future__ import annotations

from typing import Generator, Hashable, Sequence

import numpy as np
import numpy.typing as npt

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


def string_preprocessor(a: str) -> str:
    return a


class MyClass:
    def __init__(self) -> None:
        self.a: str = ""


def valid_scorer_seq1(s1: Sequence[Hashable], s2: Sequence[Hashable], score_cutoff: float | None = None) -> float:
    return 1.0


def valid_scorer_str1(s1: str, s2: str, score_cutoff: float | None = None) -> float:
    return 1.0


def valid_scorer_str2(s1: str, s2: str, *, score_cutoff: float | None) -> float:
    return 1.0


def valid_scorer_str3(string1: str, string2: str, *, score_cutoff: float | None) -> float:
    return 1.0


def invalid_scorer_str1(s1: str, s2: str, cutoff: float | None = None) -> float:
    return 1.0


def invalid_scorer_str2(s1: str, s2: str) -> float:
    return 1.0


def invalid_scorer_str3(s1: str, s2: str, score_cutoff: int | None = None) -> float:
    return 1.0


def invalid_scorer_str4(s1: int, s2: str, score_cutoff: float | None = None) -> float:
    return 1.0


def invalid_scorer_str5(s1: str, s2: int, score_cutoff: float | None = None) -> float:
    return 1.0


def test_extractOne():
    _a: tuple[str, float, int] = process.extractOne("", [""])
    _a = process.extractOne("", [""], scorer=fuzz.ratio)
    _b: tuple[str, int, int] = process.extractOne("", [""], scorer=fuzz.ratio)  # type: ignore [assignment]
    _b = process.extractOne("", [""])  # type: ignore [assignment]
    _b = process.extractOne("", [""], scorer=Levenshtein.distance)
    _a = process.extractOne(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extractOne("", [1], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extractOne(list(""), [""], scorer=fuzz.ratio)
    _c: tuple[list[str], float, int] = process.extractOne("", [list("")], scorer=fuzz.ratio)
    _a = process.extractOne("", [""], processor=string_preprocessor)
    _a = process.extractOne(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    _c = process.extractOne("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    _d: tuple[MyClass, float, int] = process.extractOne(MyClass(), [MyClass()], processor=lambda x: x.a)
    _a = process.extractOne("", [""], scorer=valid_scorer_str1)
    _a = process.extractOne("", [""], scorer=valid_scorer_str2)
    _a = process.extractOne("", [""], scorer=valid_scorer_str3)
    _a = process.extractOne("", [""], scorer=invalid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extractOne("", [""], scorer=invalid_scorer_str2)  # type: ignore [call-overload]
    _a = process.extractOne("", [""], scorer=invalid_scorer_str3)  # type: ignore [call-overload]
    _a = process.extractOne("", [""], scorer=invalid_scorer_str4)  # type: ignore [call-overload]
    _a = process.extractOne("", [""], scorer=invalid_scorer_str5)  # type: ignore [call-overload]
    _a = process.extractOne(list(""), [""], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _c = process.extractOne("", [list("")], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extractOne(list(""), [""], scorer=valid_scorer_seq1)
    _c = process.extractOne("", [list("")], scorer=valid_scorer_seq1)


def test_extract():
    _a: list[tuple[str, float, int]] = process.extract("", [""])
    _a = process.extract("", [""], scorer=fuzz.ratio)
    _b: list[tuple[str, int, int]] = process.extract("", [""], scorer=fuzz.ratio)  # type: ignore [assignment]
    _b = process.extract("", [""])  # type: ignore [assignment]
    _b = process.extract("", [""], scorer=Levenshtein.distance)
    _a = process.extract(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extract("", [1], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extract(list(""), [""], scorer=fuzz.ratio)
    _c: list[tuple[list[str], float, int]] = process.extract("", [list("")], scorer=fuzz.ratio)
    _a = process.extract("", [""], processor=string_preprocessor)
    _a = process.extract(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    _c = process.extract("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    _d: list[tuple[MyClass, float, int]] = process.extract(MyClass(), [MyClass()], processor=lambda x: x.a)
    _a = process.extract("", [""], scorer=valid_scorer_str1)
    _a = process.extract("", [""], scorer=valid_scorer_str2)
    _a = process.extract("", [""], scorer=valid_scorer_str3)
    _a = process.extract("", [""], scorer=invalid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extract("", [""], scorer=invalid_scorer_str2)  # type: ignore [call-overload]
    _a = process.extract("", [""], scorer=invalid_scorer_str3)  # type: ignore [call-overload]
    _a = process.extract("", [""], scorer=invalid_scorer_str4)  # type: ignore [call-overload]
    _a = process.extract("", [""], scorer=invalid_scorer_str5)  # type: ignore [call-overload]
    _a = process.extract(list(""), [""], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _c = process.extract("", [list("")], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extract(list(""), [""], scorer=valid_scorer_seq1)
    _c = process.extract("", [list("")], scorer=valid_scorer_seq1)


def test_extract_iter():
    _a: Generator[tuple[str, float, int], None, None] = process.extract_iter("", [""])
    _a = process.extract_iter("", [""], scorer=fuzz.ratio)
    _b: Generator[tuple[str, int, int], None, None] = process.extract_iter("", [""], scorer=fuzz.ratio)  # type: ignore [assignment]
    _b = process.extract_iter("", [""])  # type: ignore [assignment]
    _b = process.extract_iter("", [""], scorer=Levenshtein.distance)
    _a = process.extract_iter(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extract_iter("", [1], scorer=fuzz.ratio)  # type: ignore [call-overload]
    _a = process.extract_iter(list(""), [""], scorer=fuzz.ratio)
    _c: Generator[tuple[list[str], float, int], None, None] = process.extract_iter("", [list("")], scorer=fuzz.ratio)
    _a = process.extract_iter("", [""], processor=string_preprocessor)
    _a = process.extract_iter(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    _c = process.extract_iter("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    _d: Generator[tuple[MyClass, float, int], None, None] = process.extract_iter(
        MyClass(), [MyClass()], processor=lambda x: x.a
    )
    _a = process.extract_iter("", [""], scorer=valid_scorer_str1)
    _a = process.extract_iter("", [""], scorer=valid_scorer_str2)
    _a = process.extract_iter("", [""], scorer=valid_scorer_str3)
    _a = process.extract_iter("", [""], scorer=invalid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extract_iter("", [""], scorer=invalid_scorer_str2)  # type: ignore [call-overload]
    _a = process.extract_iter("", [""], scorer=invalid_scorer_str3)  # type: ignore [call-overload]
    _a = process.extract_iter("", [""], scorer=invalid_scorer_str4)  # type: ignore [call-overload]
    _a = process.extract_iter("", [""], scorer=invalid_scorer_str5)  # type: ignore [call-overload]
    _a = process.extract_iter(list(""), [""], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _c = process.extract_iter("", [list("")], scorer=valid_scorer_str1)  # type: ignore [call-overload]
    _a = process.extract_iter(list(""), [""], scorer=valid_scorer_seq1)
    _c = process.extract_iter("", [list("")], scorer=valid_scorer_seq1)


def test_cdist():
    _a: npt.NDArray[np.float32] = process.cdist([""], [""])
    _a = process.cdist([""], [""], scorer=fuzz.ratio)
    # todo these should be be rejected, but I didn't find a way to properly type the return type so far
    # _b: npt.NDArray[np.int32] = process.cdist([""], [""], scorer=fuzz.ratio)  type: ignore [assignment]
    # _b = process.cdist([""], [""])  type: ignore [assignment]
    _b: npt.NDArray[np.int32] = process.cdist([""], [""], scorer=Levenshtein.distance)
    _a = process.cdist([1], [""], scorer=fuzz.ratio)  # type: ignore [list-item]
    _a = process.cdist([""], [1], scorer=fuzz.ratio)  # type: ignore [list-item]
    _a = process.cdist([list("")], [""], scorer=fuzz.ratio)
    _a = process.cdist([""], [""], processor=string_preprocessor)
    _a = process.cdist([list("")], [""], processor=string_preprocessor)  # type: ignore [arg-type]
    _a = process.cdist([""], [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    _a = process.cdist([MyClass()], [MyClass()], processor=lambda x: x.a)
    _a = process.cdist([""], [""], scorer=valid_scorer_str1)
    _a = process.cdist([""], [""], scorer=valid_scorer_str2)
    _a = process.cdist([""], [""], scorer=valid_scorer_str3)
    _a = process.cdist([""], [""], scorer=invalid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cdist([""], [""], scorer=invalid_scorer_str2)  # type: ignore [arg-type]
    _a = process.cdist([""], [""], scorer=invalid_scorer_str3)  # type: ignore [arg-type]
    _a = process.cdist([""], [""], scorer=invalid_scorer_str4)  # type: ignore [arg-type]
    _a = process.cdist([""], [""], scorer=invalid_scorer_str5)  # type: ignore [arg-type]
    _a = process.cdist([list("")], [""], scorer=valid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cdist([""], [list("")], scorer=valid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cdist([list("")], [""], scorer=valid_scorer_seq1)
    _a = process.cdist([""], [list("")], scorer=valid_scorer_seq1)


def test_cpdist():
    _a: npt.NDArray[np.float32] = process.cpdist([""], [""])
    _a = process.cpdist([""], [""], scorer=fuzz.ratio)
    # todo these should be be rejected, but I didn't find a way to properly type the return type so far
    # _b: npt.NDArray[np.int32] = process.cdist([""], [""], scorer=fuzz.ratio)  type: ignore [assignment]
    # _b = process.cdist([""], [""])  type: ignore [assignment]
    _b: npt.NDArray[np.int32] = process.cpdist([""], [""], scorer=Levenshtein.distance)
    _a = process.cpdist([1], [""], scorer=fuzz.ratio)  # type: ignore [list-item]
    _a = process.cpdist([""], [1], scorer=fuzz.ratio)  # type: ignore [list-item]
    _a = process.cpdist([list("")], [""], scorer=fuzz.ratio)
    _a = process.cpdist([""], [""], processor=string_preprocessor)
    _a = process.cpdist([list("")], [""], processor=string_preprocessor)  # type: ignore [arg-type]
    _a = process.cpdist([""], [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    _a = process.cpdist([MyClass()], [MyClass()], processor=lambda x: x.a)
    _a = process.cpdist([""], [""], scorer=valid_scorer_str1)
    _a = process.cpdist([""], [""], scorer=valid_scorer_str2)
    _a = process.cpdist([""], [""], scorer=valid_scorer_str3)
    _a = process.cpdist([""], [""], scorer=invalid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cpdist([""], [""], scorer=invalid_scorer_str2)  # type: ignore [arg-type]
    _a = process.cpdist([""], [""], scorer=invalid_scorer_str3)  # type: ignore [arg-type]
    _a = process.cpdist([""], [""], scorer=invalid_scorer_str4)  # type: ignore [arg-type]
    _a = process.cpdist([""], [""], scorer=invalid_scorer_str5)  # type: ignore [arg-type]
    _a = process.cpdist([list("")], [""], scorer=valid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cpdist([""], [list("")], scorer=valid_scorer_str1)  # type: ignore [arg-type]
    _a = process.cpdist([list("")], [""], scorer=valid_scorer_seq1)
    _a = process.cpdist([""], [list("")], scorer=valid_scorer_seq1)
