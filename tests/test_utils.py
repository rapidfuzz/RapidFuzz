from rapidfuzz import utils


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
        assert utils.default_process(string) == proc_string
