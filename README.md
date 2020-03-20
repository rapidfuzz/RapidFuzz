<h1 align="center">
<img src="https://raw.githubusercontent.com/rhasspy/rapidfuzz/master/.github/RapidFuzz.svg?sanitize=true" alt="RapidFuzz" width="400">
</h1>
<h4 align="center">Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance</h4>

<p align="center">
  <a href="https://github.com/rhasspy/rapidfuzz/actions">
    <img src="https://github.com/rhasspy/rapidfuzz/workflows/Python%20package/badge.svg"
         alt="Continous Integration">
  </a>
  <a href="https://pypi.org/project/rapidfuzz/">
    <img src="https://img.shields.io/pypi/v/rapidfuzz"
         alt="PyPI package version">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/pypi/pyversions/rapidfuzz"
         alt="Python versions">
  </a>
  <a href="https://github.com/rhasspy/rapidfuzz/blob/dev/LICENSE">
    <img src="https://img.shields.io/github/license/rhasspy/rapidfuzz"
         alt="GitHub license">
  </a>
</p>

<p align="center">
  <a href="#why-should-you-care">Why Should You Care?</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#license">License</a>
</p>

---

## Why Should You Care?
Since there is already [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) that implements the same string similarity calculations you might wonder why you would want to use RapidFuzz. There are mainly two reasons:
1) It is MIT licensed so in contrast to FuzzyWuzzy it can be used in projects where you do not want to adopt the GPL License
2) While FuzzyWuzzy only uses python-Levenshtein for the levenshtein calculations and implements the other functionalities in Python, RapidFuzz's implementation is mostly written in C++ and on Top of this comes with a lot of Algorithmic improvements. This results in a 5-100x Speedup in String Matching.


## Installation
RapidFuzz can be installed using [pip](https://pypi.org/project/rapidfuzz/)
```bash
$ pip install rapidfuzz
```
it requires Python 3.5 or later and a C++ Compiler with C++17 support, which should be given on all current systems


## Usage
```console
> from rapidfuzz import fuzz
> from rapidfuzz import process
```

### Simple Ratio
```console
> fuzz.ratio("this is a test", "this is a test!")
96.55171966552734
```

### Partial Ratio
```console
> fuzz.partial_ratio("this is a test", "this is a test!")
100.0
```

### Token Sort Ratio
```console
> fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
90.90908813476562
> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
100.0
```

### Token Set Ratio
```console
> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
83.8709716796875
> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
100.0
```

### Process
currently no string preprocessing is done so your responsible for removing unwanted characters and to lowercase the strings
if thats what you want
```console
> choices = ["atlanta falcons", "new york jets", "new york giants", "dallas cowboys"]
> process.extract("new york jets", choices, limit=2)
[('new york jets', 100), ('new york giants', 78.57142639160156)]
> process.extractOne("cowboys", choices)
("dallas cowboys", 90)
```

## License
RapidFuzz is licensed under the MIT license since we believe that everyone should be able to use it without being forced to adopt our license. Thats why the library is based on an older version of fuzzywuzzy that was MIT licensed aswell.
A Fork of this old version of fuzzywuzzy can be found [here](https://github.com/rhasspy/fuzzywuzzy).
