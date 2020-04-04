<h1 align="center">
<img src="https://raw.githubusercontent.com/rhasspy/rapidfuzz/master/.github/RapidFuzz.svg?sanitize=true" alt="RapidFuzz" width="400">
</h1>
<h4 align="center">Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance</h4>

<p align="center">
  <a href="https://github.com/rhasspy/rapidfuzz/actions">
    <img src="https://github.com/rhasspy/rapidfuzz/workflows/Build/badge.svg"
         alt="Continous Integration">
  </a>
  <a href="https://pypi.org/project/rapidfuzz/">
    <img src="https://img.shields.io/pypi/v/rapidfuzz"
         alt="PyPI package version">
  </a>
  <a href="https://anaconda.org/conda-forge/rapidfuzz">
    <img src="https://img.shields.io/conda/vn/conda-forge/rapidfuzz.svg"
         alt="Conda Version">
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
  <a href="#description">Description</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#license">License</a>
</p>

---

## Description
RapidFuzz is a fast string matching library for Python and C++, which is using the string similarity calculations from [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy). However there are two aspects that set RapidFuzz apart from FuzzyWuzzy:
1) It is MIT licensed so it can be used whichever License you might want to choose for your project, while you're forced to adopt the GPLv2 license when using FuzzyWuzzy
2) It is mostly written in C++ and on top of this comes with a lot of Algorithmic improvements to make string matching even faster, while still providing the same results. These changes result in a 2-100x Speedup in String Matching. More details on benchmark results can be found [here](https://github.com/rhasspy/rapidfuzz/blob/master/Benchmarks.md)


## Installation
RapidFuzz can be installed using [pip](https://pypi.org/project/rapidfuzz/)
```bash
$ pip install rapidfuzz
```

We currently have pre-built binaries (wheels) for `RapidFuzz` and its dependencies for MacOS (10.9 and later), Linux x86_64 and Windows.

For any other architecture/os `RapidFuzz` can be installed from the source distribution. To do so, a C++14 capable compiler must be installed before running the `pip install rapidfuzz` command. While Linux and MacOs usually come with a compiler it is required to install [C++-Buildtools](https://visualstudio.microsoft.com/visual-cpp-build-tools) on Windows.


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
```console
> choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
> process.extract("new york jets", choices, limit=2)
[('new york jets', 100), ('new york giants', 78.57142639160156)]
> process.extractOne("cowboys", choices)
("dallas cowboys", 90)
```

## License
RapidFuzz is licensed under the MIT license since we believe that everyone should be able to use it without being forced to adopt our license. Thats why the library is based on an older version of fuzzywuzzy that was MIT licensed as well.
A Fork of this old version of fuzzywuzzy can be found [here](https://github.com/rhasspy/fuzzywuzzy).
