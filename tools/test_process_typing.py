from __future__ import annotations

from typing import Generator

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


def string_preprocessor(a: str) -> str:
    return a


class MyClass:
    def __init__(self) -> None:
        self.a: str = ""


def test_extractOne():
    a: tuple[str, float, int] = process.extractOne("", [""])
    a = process.extractOne("", [""], scorer=fuzz.ratio)
    b: tuple[str, int, int] = process.extractOne("", [""], scorer=fuzz.ratio)  # type: ignore [arg-type]
    b = process.extractOne("", [""], scorer=Levenshtein.distance)
    a = process.extractOne(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    a = process.extractOne("", [1], scorer=fuzz.ratio)  # type: ignore [list-item]
    a = process.extractOne(list(""), [""], scorer=fuzz.ratio)
    c: tuple[list[str], float, int] = process.extractOne("", [list("")], scorer=fuzz.ratio)
    a = process.extractOne("", [""], processor=string_preprocessor)
    a = process.extractOne(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    c = process.extractOne("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    d: tuple[MyClass, float, int] = process.extractOne(MyClass(), [MyClass()], processor=lambda x: x.a)


def test_extract():
    a: list[tuple[str, float, int]] = process.extract("", [""])
    a = process.extract("", [""], scorer=fuzz.ratio)
    b: list[tuple[str, int, int]] = process.extract("", [""], scorer=fuzz.ratio)  # type: ignore [arg-type]
    b = process.extract("", [""], scorer=Levenshtein.distance)
    a = process.extract(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    a = process.extract("", [1], scorer=fuzz.ratio)  # type: ignore [list-item]
    a = process.extract(list(""), [""], scorer=fuzz.ratio)
    c: list[tuple[list[str], float, int]] = process.extract("", [list("")], scorer=fuzz.ratio)
    a = process.extract("", [""], processor=string_preprocessor)
    a = process.extract(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    c = process.extract("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    d: list[tuple[MyClass, float, int]] = process.extract(MyClass(), [MyClass()], processor=lambda x: x.a)


def test_extract_iter():
    a: Generator[tuple[str, float, int], None, None] = process.extract_iter("", [""])
    a = process.extract_iter("", [""], scorer=fuzz.ratio)
    b: Generator[tuple[str, int, int], None, None] = process.extract_iter("", [""], scorer=fuzz.ratio)  # type: ignore [arg-type]
    b = process.extract_iter("", [""], scorer=Levenshtein.distance)
    a = process.extract_iter(1, [""], scorer=fuzz.ratio)  # type: ignore [call-overload]
    a = process.extract_iter("", [1], scorer=fuzz.ratio)  # type: ignore [list-item]
    a = process.extract_iter(list(""), [""], scorer=fuzz.ratio)
    c: Generator[tuple[list[str], float, int], None, None] = process.extract_iter("", [list("")], scorer=fuzz.ratio)
    a = process.extract_iter("", [""], processor=string_preprocessor)
    a = process.extract_iter(list(""), [""], processor=string_preprocessor)  # type: ignore [arg-type]
    c = process.extract_iter("", [list("")], processor=string_preprocessor)  # type: ignore [arg-type]
    d: Generator[tuple[MyClass, float, int], None, None] = process.extract_iter(
        MyClass(), [MyClass()], processor=lambda x: x.a
    )
