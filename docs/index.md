---
template: overrides/main.html
---

<h1 align="center">
<img src="img/RapidFuzz.svg" alt="RapidFuzz" width="400">
</h1>
<h4 align="center">Rapid fuzzy string matching using the Levenshtein Distance</h4>

<p align="center">
  <a href="https://pypi.org/project/rapidfuzz/">
    <img src="https://img.shields.io/pypi/v/rapidfuzz"
         alt="PyPI package version">
  </a>
  <a href="https://anaconda.org/conda-forge/rapidfuzz">
    <img src="https://img.shields.io/conda/vn/conda-forge/rapidfuzz.svg"
         alt="Conda Version">
  </a>
  <a href="https://gitter.im/rapidfuzz/community">
    <img src="https://badges.gitter.im/rapidfuzz/community.svg"
         alt="Gitter chat">
  </a>
  <a href="https://github.com/maxbachmann/rapidfuzz/blob/dev/LICENSE">
    <img src="https://img.shields.io/github/license/maxbachmann/rapidfuzz"
         alt="GitHub license">
  </a>
</p>

---

RapidFuzz is a fast string matching library for Python and C++, which is using the string similarity calculations from [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy). However there are two aspects that set RapidFuzz apart from FuzzyWuzzy:

1) It is MIT licensed so it can be used whichever License you might want to choose for your project, while you're forced to adopt the GPL license when using FuzzyWuzzy

2) It is mostly written in C++ and on top of this comes with a lot of Algorithmic improvements to make string matching even faster, while still providing the same results. More details on these performance improvements in form of benchmarks can be found [here](https://github.com/maxbachmann/rapidfuzz/blob/master/Benchmarks.md)