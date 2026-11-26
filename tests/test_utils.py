from __future__ import annotations

import pytest

from rapidfuzz import (
    fuzz_cpp,
    fuzz_py,
    process_cpp,
    process_py,
    utils,
    utils_cpp,
    utils_py,
)


def test_fullProcess():
    mixed_strings = [
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "C'est la vie",
        "Ça va?",
        "Cães danados",
        "¬Camarões assados",
        "a¬ሴ€耀",
        "Á",
    ]
    mixed_strings_proc = [
        "lorem ipsum is simply dummy text of the printing and typesetting industry",
        "c est la vie",
        "ça va",
        "cães danados",
        "camarões assados",
        "a ሴ 耀",
        "á",
    ]

    for string, proc_string in zip(mixed_strings, mixed_strings_proc):
        assert utils_cpp.default_process(string) == proc_string
        assert utils_py.default_process(string) == proc_string


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("කුරුණෑගල", "කුරුණෑගල"),
        ("தமிழ் மொழி", "தமிழ் மொழி"),
        ("राजधानी", "राजधानी"),
        ("ශ්\u200dරී ලංකාව", "ශ්\u200dරී ලංකාව"),
        ("ද\u200d්ධ", "ද\u200d්ධ"),
        ("می\u200cروم", "می\u200cروم"),
        ("👩\u200d💻", ""),
        ("\u200dabc\u200c", "abc"),
        ("a\u200d!b", "a  b"),
        ("!İ_ΟΣ?", "i\u0307 ος"),
        ("A\u20dd", "a\u20dd"),
        ("\u0301", "\u0301"),
        ("cafe\u0301", "cafe\u0301"),
        ("café", "café"),
        (" \t! ", ""),
    ],
)
def test_default_process_with_marks(text, expected):
    assert utils.default_process_with_marks(text) == expected


@pytest.mark.parametrize("value", [None, b"abc"])
def test_default_process_with_marks_requires_text(value):
    with pytest.raises(TypeError, match="sentence must be a string"):
        utils.default_process_with_marks(value)


@pytest.mark.parametrize(
    ("process_backend", "fuzz_backend"),
    [(process_py, fuzz_py), (process_cpp, fuzz_cpp)],
    ids=["python", "cpp"],
)
@pytest.mark.parametrize(
    ("query", "unrelated"),
    [("කුරුණෑගල", "කොළඹ"), ("கொழும்பு", "மட்டக்களப்பு")],
)
def test_mark_preserving_processor_in_extract(process_backend, fuzz_backend, query, unrelated):
    choices = [query, unrelated]
    options = {"scorer": fuzz_backend.WRatio, "score_cutoff": 80}
    legacy_matches = process_backend.extract(query, choices, processor=utils_py.default_process, **options)
    assert [choice for choice, _, _ in legacy_matches] == choices

    matches = process_backend.extract(query, choices, processor=utils.default_process_with_marks, **options)
    assert matches == [(query, 100.0, 0)]


@pytest.mark.parametrize("process_backend", [process_py, process_cpp], ids=["python", "cpp"])
def test_mark_preserving_processor_keeps_sinhala_conjunct_together(process_backend):
    query = "ශ්\u200dරී ලංකාව"
    matches = process_backend.extract(
        query, [query, "රී ලංකාව"], processor=utils.default_process_with_marks, score_cutoff=90
    )
    assert matches == [(query, 100.0, 0)]
