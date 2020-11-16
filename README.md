<h1 align="center">
<img src="https://raw.githubusercontent.com/maxbachmann/rapidfuzz/master/docs/img/RapidFuzz.svg?sanitize=true" alt="RapidFuzz" width="400">
</h1>
<h4 align="center">Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance</h4>

<p align="center">
  <a href="https://github.com/maxbachmann/rapidfuzz/actions">
    <img src="https://github.com/maxbachmann/rapidfuzz/workflows/Build/badge.svg"
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
  </a><br/>
  <a href="https://gitter.im/rapidfuzz/community">
    <img src="https://badges.gitter.im/rapidfuzz/community.svg"
         alt="Gitter chat">
  </a>
  <a href="https://maxbachmann.github.io/rapidfuzz">
    <img src="https://img.shields.io/badge/-documentation-blue"
         alt="Documentation">
  </a>
  <a href="https://github.com/maxbachmann/rapidfuzz/blob/dev/LICENSE">
    <img src="https://img.shields.io/github/license/maxbachmann/rapidfuzz"
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
1) It is MIT licensed so it can be used whichever License you might want to choose for your project, while you're forced to adopt the GPL license when using FuzzyWuzzy
2) It is mostly written in C++ and on top of this comes with a lot of Algorithmic improvements to make string matching even faster, while still providing the same results. More details on these performance improvements in form of benchmarks can be found [here](https://github.com/maxbachmann/rapidfuzz/blob/master/Benchmarks.md)

## Requirements

- Python 2.7 or later
- On Windows the [Visual C++ 2019 redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is required

## Installation

There are several ways to install RapidFuzz, the recommended methods
are to either use `pip`(the Python package manager) or
`conda` (an open-source, cross-platform, package manager)

### with pip

RapidFuzz can be installed with `pip` the following way:

```bash
pip install rapidfuzz
```

There are pre-built binaries (wheels) of RapidFuzz for MacOS (10.9 and later), Linux x86_64 and Windows. Wheels for armv6l (Raspberry Pi Zero) and armv7l (Raspberry Pi) are available on [piwheels](https://www.piwheels.org/project/rapidfuzz/).

> :heavy_multiplication_x: &nbsp;&nbsp;**failure "ImportError: DLL load failed"**
>
> If you run into this error on Windows the reason is most likely, that the [Visual C++ 2019 redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is not installed, which is required to find C++ Libraries (The C++ 2019 version includes the 2015, 2017 and 2019 version).


### with conda

RapidFuzz can be installed with `conda`:

```bash
conda install -c conda-forge rapidfuzz
```

### from git
RapidFuzz can be installed directly from the source distribution by cloning the repository. This requires a C++14 capable compiler.

```bash
git clone https://github.com/maxbachmann/rapidfuzz.git
cd rapidfuzz
pip install .
```

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
[('New York Jets', 100, 1), ('New York Giants', 78.57142639160156, 2)]
> process.extractOne("cowboys", choices)
("Dallas Cowboys", 90, 3)
```

## License
RapidFuzz is licensed under the MIT license since I believe that everyone should be able to use it without being forced to adopt the GPL license. Thats why the library is based on an older version of fuzzywuzzy that was MIT licensed as well.
This old version of fuzzywuzzy can be found [here](https://github.com/seatgeek/fuzzywuzzy/tree/4bf28161f7005f3aa9d4d931455ac55126918df7).
