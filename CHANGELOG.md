## Changelog

### [2.0.5] - 2022-02-25
#### Fixed
- fix signed integer overflow inside hashmap implementation

### [2.0.4] - 2022-02-21
#### Fixed
- fix binary size increase due to debug symbols
- fix segmentation fault in `Levenshtein.editops`

### [2.0.3] - 2022-02-18
#### Added
- Added fuzz.partial_ratio_alignment, which returns the result of fuzz.partial_ratio
  combined with the alignment this result stems from

#### Fixed
- Fix Indel distance returning incorrect result when using score_cutoff=1, when the strings
  are not equal. This affected other scorers like fuzz.WRatio, which use the Indel distance
  as well.

### [2.0.2] - 2022-02-12
#### Fixed
- fix type hints
- Add back transpiled cython files to the sdist to simplify builds in package builders
  like FreeBSD port build or conda-forge

### [2.0.1] - 2022-02-11
#### Fixed
- fix type hints
- Indel.normalized_similarity mistakenly used the implementation of Indel.normalized_distance

### [2.0.0] - 2022-02-09
#### Added
- added C-Api which can be used to extend RapidFuzz from different Python modules using any
  programming language which allows the usage of C-Apis (C/C++/Rust)
- added new scorers in `rapidfuzz.distance.*`
  - port existing distances to this new api
  - add Indel distance along with the corresponding editops function

#### Changed
- when the result of `string_metric.levenshtein` or `string_metric.hamming` is below max
  they do now return `max + 1` instead of -1
- Build system moved from setuptools to scikit-build
- Stop including all modules in \_\_init\_\_.py, since they significantly slowed down import time

#### Removed
- remove the `rapidfuzz.levenshtein` module which was deprecated in v1.0.0 and scheduled for removal in v2.0.0
- dropped support for Python2.7 and Python3.5

#### Deprecated
- deprecate support to specify processor in form of a boolean (will be removed in v3.0.0)
  - new functions will not get support for this in the first place
- deprecate `rapidfuzz.string_metric` (will be removed in v3.0.0). Similar scorers are available
  in `rapidfuzz.distance.*`

#### Fixed
- process.cdist did raise an exception when used with a pure python scorer

#### Performance
- improve performance and memory usage of `rapidfuzz.string_metric.levenshtein_editops`
  - memory usage is reduced by 33%
  - performance is improved by around 10%-20%
- significantly improve performance of  `rapidfuzz.string_metric.levenshtein` for `max <= 31`
  using a banded implementation

### [1.9.1] - 2021-12-13
#### Fixed
- fix bug in new editops implementation, causing it to SegFault on some inputs (see qurator-spk/dinglehopper#64)

### [1.9.0] - 2021-12-11
#### Fixed
- Fix some issues in the type annotations (see #163)

#### Performance
- improve performance and memory usage of `rapidfuzz.string_metric.levenshtein_editops`
  - memory usage is reduced by 10x
  - performance is improved from `O(N * M)` to `O([N / 64] * M)`

### [1.8.3] - 2021-11-19
#### Added
- Added missing wheels for Python3.6 on MacOs and Windows (see #159)

### [1.8.2] - 2021-10-27
#### Added
- Add wheels for Python 3.10 on MacOs

### [1.8.1] - 2021-10-22
#### Fixed
- Fix incorrect editops results (See #148)

### [1.8.0] - 2021-10-20
#### Changed
- Add Wheels for Python3.10 on all platforms except MacOs (see #141)
- Improve performance of `string_metric.jaro_similarity` and  `string_metric.jaro_winkler_similarity` for strings with a length <= 64

### [1.7.1] - 2021-10-02
#### Fixed
- fixed incorrect results of fuzz.partial_ratio for long needles (see #138)

### [1.7.0] - 2021-09-27
#### Changed
- Added typing for process.cdist
- Added multithreading support to cdist using the argument `process.cdist`
- Add dtype argument to `process.cdist` to set the dtype of the result numpy array (see #132)
- Use a better hash collision strategy in the internal hashmap, which improves the worst case performance

### [1.6.2] - 2021-09-15
#### Changed
- improved performance of fuzz.ratio
- only import process.cdist when numpy is available

### [1.6.1] - 2021-09-11
#### Changed
- Add back wheels for Python2.7

### [1.6.0] - 2021-09-10
#### Changed
- fuzz.partial_ratio uses a new implementation for short needles (<= 64). This implementation is
  - more accurate than the current implementation (it is guaranteed to find the optimal alignment)
  - it is significantly faster
- Add process.cdist to compare all elements of two lists (see #51)

### [1.5.1] - 2021-09-01
#### Fixed
- Fix out of bounds access in levenshtein_editops

### [1.5.0] - 2021-08-21
#### Changed
- all scorers do now support similarity/distance calculations between any sequence of hashables. So it is possible to calculate e.g. the WER as:
```
>>> string_metric.levenshtein(["word1", "word2"], ["word1", "word3"])
1
```

#### Added
- Added type stub files for all functions
- added jaro similarity in `string_metric.jaro_similarity`
- added jaro winkler similarity in `string_metric.jaro_winkler_similarity`
- added Levenshtein editops in `string_metric.levenshtein_editops`

#### Fixed
- Fixed support for set objects in `process.extract`
- Fixed inconsistent handling of empty strings

### [1.4.1] - 2021-03-30
#### Performance
-  improved performance of result creation in process.extract

#### Fixed
- Cython ABI stability issue (#95)
- fix missing decref in case of exceptions in process.extract

### [1.4.0] - 2021-03-29
#### Changed
- added processor support to `levenshtein` and `hamming`
- added distance support to extract/extractOne/extract_iter

#### Fixed
- incorrect results of `normalized_hamming` and `normalized_levenshtein` when used with `utils.default_process` as processor

### [1.3.3] - 2021-03-20
#### Fixed
- Fix a bug in the mbleven implementation of the uniform Levenshtein distance and cover it with fuzz tests

### [1.3.2] - 2021-03-20
#### Fixed
- some of the newly activated warnings caused build failures in the conda-forge build

### [1.3.1] - 2021-03-20
#### Fixed
- Fixed issue in LCS calculation for partial_ratio (see #90)
- Fixed incorrect results for normalized_hamming and normalized_levenshtein when the processor `utils.default_process` is used
- Fix many compiler warnings

### [1.3.0] - 2021-03-16
#### Changed
- add wheels for a lot of new platforms
- drop support for Python 2.7

#### Performance
- use `is` instead of `==` to compare functions directly by address

#### Fixed
- Fix another ref counting issue
- Fix some issues in the Levenshtein distance algorithm (see #92)

### [1.2.1] - 2021-03-08
#### Performance
- further improve bitparallel implementation of uniform Levenshtein distance for strings with a length > 64 (in many cases more than 50% faster)

### [1.2.0] - 2021-03-07
#### Changed
- add more benchmarks to documentation

#### Performance
- add bitparallel implementation to InDel Distance (Levenshtein with the weights 1,1,2) for strings with a length > 64
- improve bitparallel implementation of uniform Levenshtein distance for strings with a length > 64
- use the InDel Distance and uniform Levenshtein distance in more cases instead of the generic implementation
- Directly use the Levenshtein implementation in C++ instead of using it through Python in process.*

### [1.1.2] - 2021-03-03
#### Fixed
- Fix reference counting in process.extract (see #81)

### [1.1.1] - 2021-02-23
#### Fixed
- Fix result conversion in process.extract (see #79)

### [1.1.0] - 2021-02-21
#### Changed
- string_metric.normalized_levenshtein supports now all weights
- when different weights are used for Insertion and Deletion the strings are not swapped inside the Levenshtein implementation anymore. So different weights for Insertion and Deletion are now supported.
- replace C++ implementation with a Cython implementation. This has the following advantages:
  - The implementation is less error prone, since a lot of the complex things are done by Cython
  - slighly faster than the current implementation (up to 10% for some parts)
  - about 33% smaller binary size
  - reduced compile time
- Added **kwargs argument to process.extract/extractOne/extract_iter that is passed to the scorer
- Add max argument to hamming distance
- Add support for whole Unicode range to utils.default_process

#### Performance
- replaced Wagner Fischer usage in the normal Levenshtein distance with a bitparallel implementation

### [1.0.2] - 2021-02-19
#### Fixed
- The bitparallel LCS algorithm in fuzz.partial_ratio did not find the longest common substring properly in some cases.
The old algorithm is used again until this bug is fixed.

### [1.0.1] - 2021-02-17
#### Changed
- string_metric.normalized_levenshtein supports now the weights (1, 1, N) with N >= 1

#### Performance
- The Levenshtein distance with the weights (1, 1, >2) do now use the same implementation as the weight (1, 1, 2), since
  `Substitution > Insertion + Deletion` has no effect

#### Fixed
- fix uninitialized variable in bitparallel Levenshtein distance with the weight (1, 1, 1)

### [1.0.0] - 2021-02-12
#### Changed
- all normalized string_metrics can now be used as scorer for process.extract/extractOne
- Implementation of the C++ Wrapper completely refactored to make it easier to add more scorers, processors and string matching algorithms in the future.
- increased test coverage, that already helped to fix some bugs and help to prevent regressions in the future
- improved docstrings of functions

#### Performance
- Added bit-parallel implementation of the Levenshtein distance for the weights (1,1,1) and (1,1,2).
- Added specialized implementation of the Levenshtein distance for cases with a small maximum edit distance, that is even faster, than the bit-parallel implementation.
- Improved performance of `fuzz.partial_ratio`
-> Since `fuzz.ratio` and `fuzz.partial_ratio` are used in most scorers, this improves the overall performance.
- Improved performance of `process.extract` and `process.extractOne`

#### Deprecated
- the `rapidfuzz.levenshtein` module is now deprecated and will be removed in v2.0.0
  These functions are now placed in `rapidfuzz.string_metric`. `distance`, `normalized_distance`, `weighted_distance` and `weighted_normalized_distance` are combined into `levenshtein` and `normalized_levenshtein`.

#### Added
- added normalized version of the hamming distance in `string_metric.normalized_hamming`
- process.extract_iter as a generator, that yields the similarity of all elements, that have a similarity >= score_cutoff

#### Fixed
- multiple bugs in extractOne when used with a scorer, that's not from RapidFuzz
- fixed bug in `token_ratio`
- fixed bug in result normalization causing zero division


### [0.14.2] - 2020-12-31
#### Fixed
- utf8 usage in the copyright header caused problems with python2.7 on some platforms (see #70)

### [0.14.1] - 2020-12-13
#### Fixed
- when a custom processor like `lambda s: s` was used with any of the methods inside fuzz.* it always returned a score of 100. This release fixes this and adds a better test coverage to prevent this bug in the future.

### [0.14.0] - 2020-12-09
#### Added
- added hamming distance metric in the levenshtein module

#### Performance
- improved performance of default_process by using lookup table

### [0.13.4] - 2020-11-30
#### Fixed
- Add missing virtual destructor that caused a segmentation fault on Mac Os

### [0.13.3] - 2020-11-21
#### Added
- C++11 Support
- manylinux wheels

### [0.13.2] - 2020-11-21
#### Fixed
- Levenshtein was not imported from \_\_init\_\_
- The reference count of a Python Object inside process.extractOne was decremented to early

### [0.13.1] - 2020-11-17
#### Performance
- process.extractOne  exits early when a score of 100 is found. This way the other strings do not have to be preprocessed anymore.

### [0.13.0] - 2020-11-16
#### Fixed
- string objects passed to scorers had to be strings even before preprocessing them. This was changed, so they only have to be strings after preprocessing similar to process.extract/process.extractOne

#### Performance
- process.extractOne is now implemented in C++ making it a lot faster
- When token_sort_ratio or partial_token_sort ratio is used inprocess.extractOne the words in the query are only sorted once to improve the runtime

#### Changed
- process.extractOne/process.extract do now return the index of the match, when the choices are a list.

#### Removed
- process.extractIndices got removed, since the indices are now already returned by process.extractOne/process.extract

### [0.12.5] - 2020-10-26
#### Fixed
- fix documentation of process.extractOne (see #48)

### [0.12.4] - 2020-10-22
#### Added
- Added wheels for
  - CPython 2.7 on windows 64 bit
  - CPython 2.7 on windows 32 bit
  - PyPy 2.7 on windows 32 bit

### [0.12.3] - 2020-10-09
#### Fixed
- fix bug in partial_ratio (see #43)

### [0.12.2] - 2020-10-01
#### Fixed
- fix inconsistency with fuzzywuzzy in partial_ratio when using strings of equal length

### [0.12.1] - 2020-09-30
#### Fixed
- MSVC has a bug and therefore crashed on some of the templates used. This Release simplifies the templates so compiling on msvc works again

### [0.12.0] - 2020-09-30
#### Performance
- partial_ratio is using the Levenshtein distance now, which is a lot faster. Since many of the other algorithms use partial_ratio, this helps to improve the overall performance

### [0.11.3] - 2020-09-22
#### Fixed
- fix partial_token_set_ratio returning 100 all the time

### [0.11.2] - 2020-09-12
#### Added
- added rapidfuzz.\_\_author\_\_, rapidfuzz.\_\_license\_\_ and rapidfuzz.\_\_version\_\_ 

### [0.11.1] - 2020-09-01
#### Fixed
- do not use auto junk when searching the optimal alignment for partial_ratio

### [0.11.0] - 2020-08-22
#### Changed
- support for python 2.7 added #40 
- add wheels for python2.7 (both pypy and cpython) on MacOS and Linux

### [0.10.0] - 2020-08-17
#### Changed
- added wheels for Python3.9

#### Fixed
- tuple scores in process.extractOne are now supported #39
