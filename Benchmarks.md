# Benchmarks

To compare the speed of FuzzyWuzzy and RapidFuzz the Benchmark of FuzzyWuzzy is used.
Therefore the Benchmark is always executed for FuzzyWuzzy, RapidFuzz and when directly calling the CPP handler without redirection (e.g. `rapidfuzz.fuzz.fuzz_cpp.ratio`).
Afterwards a ratio between the runtime of both results is calculated. The benchmark can be found [here](https://github.com/rhasspy/rapidfuzz/blob/master/python/bench). The results of the benchmarks are visualised below.

## fuzz.ratio

<img src="https://raw.githubusercontent.com/rhasspy/rapidfuzz/master/.github/fuzz.ratio.svg?sanitize=true">


## fuzz.partial_ratio

<img src="https://raw.githubusercontent.com/rhasspy/rapidfuzz/master/.github/fuzz.partial_ratio.svg?sanitize=true">


## fuzz.WRatio

<img src="https://raw.githubusercontent.com/rhasspy/rapidfuzz/master/.github/fuzz.WRatio.svg?sanitize=true">
