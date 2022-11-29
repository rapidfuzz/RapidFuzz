import pytest

from rapidfuzz import fuzz, process_cpp, process_py

def wrapped(func):
    from functools import wraps
    @wraps(func)
    def decorator(*args, **kwargs):
        return 100

    return decorator

class process:
    @staticmethod
    def extract_iter(*args, **kwargs):
        res1 = process_cpp.extract_iter(*args, **kwargs)
        res2 = process_py.extract_iter(*args, **kwargs)

        for elem1, elem2 in zip(res1, res2, strict=True):
            assert elem1 == elem2
            yield elem1

    @staticmethod
    def extractOne(*args, **kwargs):
        res1 = process_cpp.extractOne(*args, **kwargs)
        res2 = process_py.extractOne(*args, **kwargs)
        assert res1 == res2
        return res1

    @staticmethod
    def extract(*args, **kwargs):
        res1 = process_cpp.extract(*args, **kwargs)
        res2 = process_py.extract(*args, **kwargs)
        assert res1 == res2
        return res1

    @staticmethod
    def cdist(*args, **kwargs):
        res1 = process_cpp.cdist(*args, **kwargs)
        res2 = process_py.cdist(*args, **kwargs)
        assert res1.dtype == res2.dtype
        assert res1.shape == res2.shape
        if res1.size and res2.size:
            assert res1 == res2
        return res1


baseball_strings = [
    "new york mets vs chicago cubs",
    "chicago cubs vs chicago white sox",
    "philladelphia phillies vs atlanta braves",
    "braves vs mets",
]


def test_extractOne_exceptions():
    with pytest.raises(TypeError):
        process_cpp.extractOne()
    with pytest.raises(TypeError):
        process_py.extractOne()
    with pytest.raises(TypeError):
        process_cpp.extractOne(1)
    with pytest.raises(TypeError):
        process_py.extractOne(1)
    with pytest.raises(TypeError):
        process_cpp.extractOne(1, [])
    with pytest.raises(TypeError):
        process_py.extractOne(1, [])
    with pytest.raises(TypeError):
        process_cpp.extractOne("", [1])
    with pytest.raises(TypeError):
        process_py.extractOne("", [1])
    with pytest.raises(TypeError):
        process_cpp.extractOne("", {1: 1})
    with pytest.raises(TypeError):
        process_py.extractOne("", {1: 1})


def test_extract_exceptions():
    with pytest.raises(TypeError):
        process_cpp.extract()
    with pytest.raises(TypeError):
        process_py.extract()
    with pytest.raises(TypeError):
        process_cpp.extract(1)
    with pytest.raises(TypeError):
        process_py.extract(1)
    with pytest.raises(TypeError):
        process_cpp.extract(1, [])
    with pytest.raises(TypeError):
        process_py.extract(1, [])
    with pytest.raises(TypeError):
        process_cpp.extract("", [1])
    with pytest.raises(TypeError):
        process_py.extract("", [1])
    with pytest.raises(TypeError):
        process_cpp.extract("", {1: 1})
    with pytest.raises(TypeError):
        process_py.extract("", {1: 1})


def test_extract_iter_exceptions():
    with pytest.raises(TypeError):
        process_cpp.extract_iter()
    with pytest.raises(TypeError):
        process_py.extract_iter()
    with pytest.raises(TypeError):
        process_cpp.extract_iter(1)
    with pytest.raises(TypeError):
        process_py.extract_iter(1)
    with pytest.raises(TypeError):
        next(process_cpp.extract_iter(1, []))
    with pytest.raises(TypeError):
        next(process_py.extract_iter(1, []))
    with pytest.raises(TypeError):
        next(process_cpp.extract_iter("", [1]))
    with pytest.raises(TypeError):
        next(process_py.extract_iter("", [1]))
    with pytest.raises(TypeError):
        next(process_cpp.extract_iter("", {1: 1}))
    with pytest.raises(TypeError):
        next(process_py.extract_iter("", {1: 1}))


def test_get_best_choice1():
    query = "new york mets at atlanta braves"
    best = process.extractOne(query, baseball_strings)
    assert best[0] == "braves vs mets"
    best = process.extractOne(query, set(baseball_strings))
    assert best[0] == "braves vs mets"

    best = process.extract(query, baseball_strings)[0]
    assert best[0] == "braves vs mets"
    best = process.extract(query, set(baseball_strings))[0]
    assert best[0] == "braves vs mets"


def test_get_best_choice2():
    query = "philadelphia phillies at atlanta braves"
    best = process.extractOne(query, baseball_strings)
    assert best[0] == baseball_strings[2]
    best = process.extractOne(query, set(baseball_strings))
    assert best[0] == baseball_strings[2]

    best = process.extract(query, baseball_strings)[0]
    assert best[0] == baseball_strings[2]
    best = process.extract(query, set(baseball_strings))[0]
    assert best[0] == baseball_strings[2]


def test_get_best_choice3():
    query = "atlanta braves at philadelphia phillies"
    best = process.extractOne(query, baseball_strings)
    assert best[0] == baseball_strings[2]
    best = process.extractOne(query, set(baseball_strings))
    assert best[0] == baseball_strings[2]

    best = process.extract(query, baseball_strings)[0]
    assert best[0] == baseball_strings[2]
    best = process.extract(query, set(baseball_strings))[0]
    assert best[0] == baseball_strings[2]


def test_get_best_choice4():
    query = "chicago cubs vs new york mets"
    best = process.extractOne(query, baseball_strings)
    assert best[0] == baseball_strings[0]
    best = process.extractOne(query, set(baseball_strings))
    assert best[0] == baseball_strings[0]


def test_with_processor():
    """
    extractOne should accept any type as long as it is a string
    after preprocessing
    """
    events = [
        ["chicago cubs vs new york mets", "CitiField", "2011-05-11", "8pm"],
        ["new york yankees vs boston red sox", "Fenway Park", "2011-05-11", "8pm"],
        ["atlanta braves vs pittsburgh pirates", "PNC Park", "2011-05-11", "8pm"],
    ]
    query = events[0]

    best = process.extractOne(query, events, processor=lambda event: event[0])
    assert best[0] == events[0]


def test_with_scorer():
    choices = [
        "new york mets vs chicago cubs",
        "chicago cubs at new york mets",
        "atlanta braves vs pittsbugh pirates",
        "new york yankees vs boston red sox",
    ]

    choices_mapping = {
        1: "new york mets vs chicago cubs",
        2: "chicago cubs at new york mets",
        3: "atlanta braves vs pittsbugh pirates",
        4: "new york yankees vs boston red sox",
    }

    # in this hypothetical example we care about ordering, so we use quick ratio
    query = "new york mets at chicago cubs"

    # first, as an example, the normal way would select the "more 'complete' match of choices[1]"
    best = process.extractOne(query, choices)
    assert best[0] == choices[1]
    best = process.extract(query, choices)[0]
    assert best[0] == choices[1]
    # dict
    best = process.extractOne(query, choices_mapping)
    assert best[0] == choices_mapping[2]
    best = process.extract(query, choices_mapping)[0]
    assert best[0] == choices_mapping[2]

    # now, use the custom scorer
    best = process.extractOne(query, choices, scorer=fuzz.QRatio)
    assert best[0] == choices[0]
    best = process.extract(query, choices, scorer=fuzz.QRatio)[0]
    assert best[0] == choices[0]
    # dict
    best = process.extractOne(query, choices_mapping, scorer=fuzz.QRatio)
    assert best[0] == choices_mapping[1]
    best = process.extract(query, choices_mapping, scorer=fuzz.QRatio)[0]
    assert best[0] == choices_mapping[1]


def test_with_cutoff():
    choices = [
        "new york mets vs chicago cubs",
        "chicago cubs at new york mets",
        "atlanta braves vs pittsbugh pirates",
        "new york yankees vs boston red sox",
    ]

    query = "los angeles dodgers vs san francisco giants"

    # in this situation, this is an event that does not exist in the list
    # we don't want to randomly match to something, so we use a reasonable cutoff
    best = process.extractOne(query, choices, score_cutoff=50)
    assert best is None

    # however if we had no cutoff, something would get returned
    best = process.extractOne(query, choices)
    assert best is not None


def test_with_cutoff_edge_cases():
    choices = [
        "new york mets vs chicago cubs",
        "chicago cubs at new york mets",
        "atlanta braves vs pittsbugh pirates",
        "new york yankees vs boston red sox",
    ]

    query = "new york mets vs chicago cubs"
    # Only find 100-score cases
    best = process.extractOne(query, choices, score_cutoff=100)
    assert best is not None
    assert best[0] == choices[0]

    # 0-score cases do not return None
    best = process.extractOne("", choices)
    assert best is not None
    assert best[1] == 0


def test_none_elements():
    """
    when a None element is used, it is skipped and the index is still correct
    """
    best = process.extractOne("test", [None, "tes"])
    assert best[2] == 1
    best = process.extractOne(None, [None, "tes"])
    assert best is None

    best = process.extract("test", [None, "tes"])
    assert best[0][2] == 1
    best = process.extract(None, [None, "tes"])
    assert best == []


def test_result_order():
    """
    when multiple elements have the same score, the first one should be returned
    """
    best = process.extractOne("test", ["tes", "tes"])
    assert best[2] == 0

    best = process.extract("test", ["tes", "tes"], limit=1)
    assert best[0][2] == 0


def test_empty_strings():
    choices = [
        "",
        "new york mets vs chicago cubs",
        "new york yankees vs boston red sox",
        "",
        "",
    ]

    query = "new york mets at chicago cubs"

    best = process.extractOne(query, choices)
    assert best[0] == choices[1]


def test_none_strings():
    choices = [
        None,
        "new york mets vs chicago cubs",
        "new york yankees vs boston red sox",
        None,
        None,
    ]

    query = "new york mets at chicago cubs"

    best = process.extractOne(query, choices)
    assert best[0] == choices[1]


def test_issue81():
    # this mostly tests whether this segfaults due to incorrect ref counting
    pd = pytest.importorskip("pandas")
    choices = pd.Series(
        ["test color brightness", "test lemon", "test lavender"],
        index=[67478, 67479, 67480],
    )
    matches = process.extract("test", choices)
    assert matches == [
        ("test color brightness", 90.0, 67478),
        ("test lemon", 90.0, 67479),
        ("test lavender", 90.0, 67480),
    ]


def custom_scorer(s1, s2, processor=None, score_cutoff=0):
    return fuzz.ratio(s1, s2, processor=processor, score_cutoff=score_cutoff)


@pytest.mark.parametrize("processor", [False, None, lambda s: s])
@pytest.mark.parametrize("scorer", [fuzz.ratio, custom_scorer])
def test_extractOne_case_sensitive(processor, scorer):
    assert (
        process.extractOne(
            "new york mets",
            ["new", "new YORK mets"],
            processor=processor,
            scorer=scorer,
        )[1]
        != 100
    )


@pytest.mark.parametrize("scorer", [fuzz.ratio, custom_scorer])
def test_extractOne_use_first_match(scorer):
    assert (
        process.extractOne(
            "new york mets", ["new york mets", "new york mets"], scorer=scorer
        )[2]
        == 0
    )


@pytest.mark.parametrize("scorer", [fuzz.ratio, fuzz.WRatio, custom_scorer])
def test_cdist_empty_seq(scorer):
    pytest.importorskip("numpy")
    assert process.cdist([], ["a", "b"], scorer=scorer).shape == (0, 2)
    assert process.cdist(["a", "b"], [], scorer=scorer).shape == (2, 0)


@pytest.mark.parametrize("scorer", [fuzz.ratio])
def test_wrapped_function(scorer):
    pytest.importorskip("numpy")
    scorer = wrapped(scorer)
    assert process.cdist(["test"], [float("nan")], scorer=scorer)[0, 0] == 100
    assert process.cdist(["test"], [None], scorer=scorer)[0, 0] == 100
    assert process.cdist(["test"], ["tes"], scorer=scorer)[0, 0] == 100
