Changelog
---------

[3.9.1] - 2024-05-19
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* disable AVX2 on MacOS since it did lead to illegal instructions being generated


[3.9.0] - 2024-05-02
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* significantly improve type hints for the library

Fixed
~~~~~
* fix cmake version parsing 


[3.8.1] - 2024-04-07
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* use the correct version of ``rapidfuzz-cpp`` when building against a system installed version


[3.8.0] - 2024-04-06
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added ``process.cpdist`` which allows pairwise comparison of two collection of inputs

Fixed
~~~~~
- fix some minor errors in the type hints
- fix potentially incorrect results of JaroWinkler when using high prefix weights


[3.7.0] - 2024-03-21
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* reduce importtime

[3.6.2] - 2024-03-05
^^^^^^^^^^^^^^^^^^^^

Changed
~~~~~~~
* upgrade to ``Cython==3.0.9``

Fixed
~~~~~
* upgrade ``rapidfuzz-cpp`` which includes a fix for build issues on some compilers
* fix some issues with the sphinx config

[3.6.1] - 2023-12-28
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix overflow error on systems with ``sizeof(size_t) < 8``

[3.6.0] - 2023-12-26
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix pure python fallback implementation of ``fuzz.token_set_ratio``
* properly link with ``-latomic`` if ``std::atomic<uint64_t>`` is not natively supported

Performance
~~~~~~~~~~~
* add banded implementation of LCS / Indel. This improves the runtime from ``O((|s1|/64) * |s2|)`` to ``O((score_cutoff/64) * |s2|)``

Changed
~~~~~~~
* upgrade to ``Cython==3.0.7``
* cdist for many metrics now returns a matrix of ``uint32`` instead of ``int32`` by default

[3.5.2] - 2023-11-02
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* use _mm_malloc/_mm_free on macOS if aligned_alloc is unsupported

[3.5.1] - 2023-10-31
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix compilation failure on macOS

[3.5.0] - 2023-10-31
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* skip pandas ``pd.NA`` similar to ``None``
* add ``score_multiplier`` argument to ``process.cdist`` which allows multiplying the end result scores
  with a constant factor.
* drop support for Python 3.7

Performance
~~~~~~~~~~~
* improve performance of simd implementation for ``LCS`` / ``Indel`` / ``Jaro`` / ``JaroWinkler``
* improve performance of Jaro and Jaro Winkler for long sequences
* implement ``process.extract`` with ``limit=1`` using ``process.extractOne`` which can be faster

Fixed
~~~~~
* the preprocessing function was always called through Python due to a broken C-API version check
* fix wraparound issue in simd implementation of Jaro and Jaro Winkler

[3.4.0] - 2023-10-09
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* upgrade to ``Cython==3.0.3``
* add simd implementation for Jaro and Jaro Winkler

[3.3.1] - 2023-09-25
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* add missing tag for python 3.12 support

[3.3.0] - 2023-09-11
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* upgrade to ``Cython==3.0.2``
* implement the remaining missing features from the C++ implementation in the pure Python implementation

Added
~~~~~
* added support for Python 3.12

[3.2.0] - 2023-08-02
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* build x86 with sse2/avx2 runtime detection

[3.1.2] - 2023-07-19
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* upgrade to ``Cython==3.0.0``

[3.1.1] - 2023-06-06
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* upgrade to ``taskflow==3.6``

Fixed
~~~~~
* replace usage of ``isnan`` with ``std::isnan`` which fixes the build on NetBSD

[3.1.0] - 2023-06-02
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* added keyword argument ``pad`` to Hamming distance. This controls whether sequences of different
  length should be padded or lead to a ``ValueError``
* improve consistency of exception messages between the C++ and pure Python implementation
* upgrade required Cython version to ``Cython==3.0.0b3``

Fixed
~~~~~
* fix missing GIL restore when an exception is thrown inside ``process.cdist``
* fix incorrect type hints for the ``process`` module

[3.0.0] - 2023-04-16
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* allow the usage of ``Hamming`` for different string lengths. Length differences are handled as
  insertions / deletions
* remove support for boolean preprocessor functions in ``rapidfuzz.fuzz`` and ``rapidfuzz.process``.
  The processor argument is now always a callable or ``None``.
* update defaults of the processor argument to be ``None`` everywhere. For affected functions this can change results, since strings are no longer preprocessed.
  To get back the old behaviour pass ``processor=utils.default_process`` to these functions.
  The following functions are affected by this:

  * ``process.extract``, ``process.extract_iter``, ``process.extractOne``
  * ``fuzz.token_sort_ratio``, ``fuzz.token_set_ratio``, ``fuzz.token_ratio``, ``fuzz.partial_token_sort_ratio``, ``fuzz.partial_token_set_ratio``, ``fuzz.partial_token_ratio``, ``fuzz.WRatio``, ``fuzz.QRatio``

* ``rapidfuzz.process`` no longer calls scorers with ``processor=None``. For this reason user provided scorers no longer require this argument.
* remove option to pass keyword arguments to scorer via ``**kwargs`` in ``rapidfuzz.process``. They can be passed
  via a ``scorer_kwargs`` argument now. This ensures this does not break when extending function parameters and
  prevents naming clashes.
* remove ``rapidfuzz.string_metric`` module. Replacements for all functions are available in ``rapidfuzz.distance``

Added
~~~~~
* added support for arbitrary hashable sequence in the pure Python fallback implementation of all functions in ``rapidfuzz.distance``
* added support for ``None`` and ``float("nan")`` in ``process.cdist`` as long as the underlying scorer supports it.
  This is the case for all scorers returning normalized results.

Fixed
~~~~~
* fix division by zero in simd implementation of normalized metrics leading to incorrect results

[2.15.1] - 2023-04-11
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix incorrect tag dispatching implementation leading to AVX2 instructions in the SSE2 code path

Added
~~~~~
* add wheels for windows arm64

[2.15.0] - 2023-04-01
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* allow the usage of finite generators as choices in ``process.extract``

[2.14.0] - 2023-03-31
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* upgrade required Cython version to ``Cython==3.0.0b2``

Fixed
~~~~~
* fix handling of non symmetric scorers in pure python version of ``process.cdist``
* fix default dtype handling when using ``process.cdist`` with pure python scorers

[2.13.7] - 2022-12-20
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~~~
* fix function signature of ``get_requires_for_build_wheel``

[2.13.6] - 2022-12-11
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* reformat changelog as restructured text to get rig of ``m2r2`` dependency


[2.13.5] - 2022-12-11
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added docs to sdist

Fixed
~~~~~
* fix two cases of undefined behavior in ``process.cdist``

[2.13.4] - 2022-12-08
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* handle ``float("nan")`` similar to ``None`` for query / choice, since this is common for
  non-existent data in tools like numpy

Fixed
~~~~~
* fix handling on ``None``\ /\ ``float("nan")`` in ``process.distance``
* use absolute imports inside tests

[2.13.3] - 2022-12-03
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* improve handling of functions wrapped using ``functools.wraps``
* fix broken fallback to Python implementation when the a ``ImportError`` occurs on import.
  This can e.g. occur when the binary has a dependency on libatomic, but it is unavailable on
  the system
* define ``CMAKE_C_COMPILER_AR``\ /\ ``CMAKE_CXX_COMPILER_AR``\ /\ ``CMAKE_C_COMPILER_RANLIB``\ /\ ``CMAKE_CXX_COMPILER_RANLIB``
  if they are not defined yet

[2.13.2] - 2022-11-05
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix incorrect results in ``Hamming.normalized_similarity``
* fix incorrect score_cutoff handling in pure python implementation of
  ``Postfix.normalized_distance`` and ``Prefix.normalized_distance``
* fix ``Levenshtein.normalized_similarity`` and ``Levenshtein.normalized_distance``
  when used in combination with the process module
* ``fuzz.partial_ratio`` was not always symmetric when ``len(s1) == len(s2)``

[2.13.1] - 2022-11-02
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix bug in ``normalized_similarity`` of most scorers,
  leading to incorrect results when used in combination with the process module
* fix sse2 support
* fix bug in ``JaroWinkler`` and ``Jaro`` when used in the pure python process module
* forward kwargs in pure Python implementation of ``process.extract``

[2.13.0] - 2022-10-30
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix bug in ``Levenshtein.editops`` leading to crashes when used with ``score_hint``

Changed
~~~~~~~
* moved capi from ``rapidfuzz_capi`` into ``rapidfuzz``\ , since it will always
  succeed the installation now that there is a pure Python mode
* add ``score_hint`` argument to process module
* add ``score_hint`` argument to Levenshtein module

[2.12.0] - 2022-10-24
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* drop support for Python 3.6

Added
~~~~~
* added ``Prefix``\ /\ ``Suffix`` similarity

Fixed
~~~~~
* fixed packaging with pyinstaller

[2.11.1] - 2022-10-05
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix segmentation fault in ``process.cdist`` when used with an empty query sequence

[2.11.0] - 2022-10-02
^^^^^^^^^^^^^^^^^^^^^
Changes
~~~~~~~
* move jarowinkler dependency into rapidfuzz to simplify maintenance

Performance
~~~~~~~~~~~
* add SIMD implementation for ``fuzz.ratio``\ /\ ``fuzz.QRatio``\ /\ ``Levenshtein``\ /\ ``Indel``\ /\ ``LCSseq``\ /\ ``OSA`` to improve
  performance for short strings in cdist

[2.10.3] - 2022-09-30
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* use ``scikit-build=0.14.1`` on Linux, since ``scikit-build=0.15.0`` fails to find the Python Interpreter
* workaround gcc in bug in template type deduction

[2.10.2] - 2022-09-27
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix support for cmake versions below 3.17

[2.10.1] - 2022-09-25
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* modernize cmake build to fix most conda-forge builds

[2.10.0] - 2022-09-18
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* add editops to hamming distance

Performance
~~~~~~~~~~~
* strip common affix in osa distance

Fixed
~~~~~
* ignore missing pandas in Python 3.11 tests

[2.9.0] - 2022-09-16
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* add optimal string alignment (OSA)

[2.8.0] - 2022-09-11
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* ``fuzz.partial_ratio`` did not find the optimal alignment in some edge cases (#219)

Performance
~~~~~~~~~~~
* improve performance of ``fuzz.partial_ratio``

Changed
~~~~~~~
* increased minimum C++ version to C++17 (see #255)

[2.7.0] - 2022-09-11
^^^^^^^^^^^^^^^^^^^^
Performance
~~~~~~~~~~~
* improve performance of ``Levenshtein.distance``\ /\ ``Levenshtein.editops`` for
  long sequences.

Added
~~~~~
* add ``score_hint`` parameter to ``Levenshtein.editops`` which allows the use of a
  faster implementation

Changed
~~~~~~~
* all functions in the ``string_metric`` module do now raise a deprecation warning.
  They are now only wrappers for their replacement functions, which makes them slower
  when used with the process module

[2.6.1] - 2022-09-03
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix incorrect results of partial_ratio for long needles (#257)

[2.6.0] - 2022-08-20
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix hashing for custom classes

Added
~~~~~
* add support for slicing in ``Editops.__getitem__``\ /\ ``Editops.__delitem__``
* add ``DamerauLevenshtein`` module

[2.5.0] - 2022-08-14
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added support for KeyboardInterrupt in processor module
  It might still take a bit until the KeyboardInterrupt is registered, but
  no longer runs all text comparisons after pressing ``Ctrl + C``

Fixed
~~~~~
* fix default scorer used by cdist to use C++ implementation if possible

[2.4.4] - 2022-08-12
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* Added support for Python 3.11

[2.4.3] - 2022-08-08
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix value range of ``jaro_similarity``\ /\ ``jaro_winkler_similarity`` in the pure Python mode
  for the string_metric module
* fix missing atomix symbol on arm 32 bit

[2.4.2] - 2022-07-30
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* add missing symbol to pure Python which made the usage impossible

[2.4.1] - 2022-07-29
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix version number

[2.4.0] - 2022-07-29
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix banded Levenshtein implementation

Performance
~~~~~~~~~~~
* improve performance and memory usage of ``Levenshtein.editops``

  * memory usage is reduced from O(NM) to O(N)
  * performance is improved for long sequences

[2.3.0] - 2022-07-23
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* add ``as_matching_blocks`` to ``Editops``\ /\ ``Opcodes``
* add support for deletions from ``Editops``
* add ``Editops.apply``\ /\ ``Opcodes.apply``
* add ``Editops.remove_subsequence``

Changed
~~~~~~~
* merge adjacent similar blocks in ``Opcodes``

Fixed
~~~~~
* fix usage of ``eval(repr(Editop))``\ , ``eval(repr(Editops))``\ , ``eval(repr(Opcode))`` and ``eval(repr(Opcodes))``
* fix opcode conversion for empty source sequence
* fix validation for empty Opcode list passed into ``Opcodes.__init__``

[2.2.0] - 2022-07-19
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* added in-tree build backend to install cmake and ninja only when it is not installed yet
  and only when wheels are available

[2.1.4] - 2022-07-17
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* changed internal implementation of cdist to remove build dependency to numpy

Added
~~~~~
* added wheels for musllinux and manylinux ppc64le, s390x

[2.1.3] - 2022-07-09
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix missing type stubs

[2.1.2] - 2022-07-04
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* change src layout to make package import from root directory possible

[2.1.1] - 2022-06-30
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* allow installation without the C++ extension if it fails to compile
* allow selection of implementation via the environment variable ``RAPIDFUZZ_IMPLEMENTATION``
  which can be set to "cpp" or "python"

[2.1.0] - 2022-06-29
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added pure python fallback for all implementations with the following exceptions:

  * no support for sequences of hashables. Only strings supported so far
  * ``\*.editops`` / ``\*.opcodes`` functions not implemented yet
  * process.cdist does not support multithreading

Fixed
~~~~~
* fuzz.partial_ratio_alignment ignored the score_cutoff
* fix implementation of Hamming.normalized_similarity
* fix default score_cutoff of Hamming.similarity
* fix implementation of LCSseq.distance when used in the process module
* treat hash for -1 and -2 as different

[2.0.15] - 2022-06-24
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix integer wraparound in partial_ratio/partial_ratio_alignment

[2.0.14] - 2022-06-23
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix unlimited recursion in LCSseq when used in combination with the process module

Changed
~~~~~~~
* add fallback implementations of ``taskflow``\ , ``rapidfuzz-cpp`` and ``jarowinkler-cpp``
  back to wheel, since some package building systems like piwheels can't clone sources

[2.0.13] - 2022-06-22
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* use system version of cmake on arm platforms, since the cmake package fails to compile

[2.0.12] - 2022-06-22
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* add tests to sdist
* remove cython dependency for sdist

[2.0.11] - 2022-04-23
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* relax version requirements of dependencies to simplify packaging

[2.0.10] - 2022-04-17
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Do not include installations of jaro_winkler in wheels (regression from 2.0.7)

Changed
~~~~~~~
* Allow installation from system installed versions of ``rapidfuzz-cpp``\ , ``jarowinkler-cpp``
  and ``taskflow``

Added
~~~~~
* Added PyPy3.9 wheels on Linux

[2.0.9] - 2022-04-07
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Add missing Cython code in sdist
* consider float imprecision in score_cutoff (see #210)

[2.0.8] - 2022-04-07
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix incorrect score_cutoff handling in token_set_ratio and token_ratio

Added
~~~~~
* add longest common subsequence

[2.0.7] - 2022-03-13
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Do not include installations of jaro_winkler and taskflow in wheels

[2.0.6] - 2022-03-06
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix incorrect population of sys.modules which lead to submodules overshadowing
  other imports

Changed
~~~~~~~
* moved JaroWinkler and Jaro into a separate package

[2.0.5] - 2022-02-25
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix signed integer overflow inside hashmap implementation

[2.0.4] - 2022-02-21
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix binary size increase due to debug symbols
* fix segmentation fault in ``Levenshtein.editops``

[2.0.3] - 2022-02-18
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* Added fuzz.partial_ratio_alignment, which returns the result of fuzz.partial_ratio
  combined with the alignment this result stems from

Fixed
~~~~~
* Fix Indel distance returning incorrect result when using score_cutoff=1, when the strings
  are not equal. This affected other scorers like fuzz.WRatio, which use the Indel distance
  as well.

[2.0.2] - 2022-02-12
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix type hints
* Add back transpiled cython files to the sdist to simplify builds in package builders
  like FreeBSD port build or conda-forge

[2.0.1] - 2022-02-11
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix type hints
* Indel.normalized_similarity mistakenly used the implementation of Indel.normalized_distance

[2.0.0] - 2022-02-09
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added C-Api which can be used to extend RapidFuzz from different Python modules using any
  programming language which allows the usage of C-Apis (C/C++/Rust)
* added new scorers in ``rapidfuzz.distance.*``

  * port existing distances to this new api
  * add Indel distance along with the corresponding editops function

Changed
~~~~~~~
* when the result of ``string_metric.levenshtein`` or ``string_metric.hamming`` is below max
  they do now return ``max + 1`` instead of -1
* Build system moved from setuptools to scikit-build
* Stop including all modules in __init__.py, since they significantly slowed down import time

Removed
~~~~~~~
* remove the ``rapidfuzz.levenshtein`` module which was deprecated in v1.0.0 and scheduled for removal in v2.0.0
* dropped support for Python2.7 and Python3.5

Deprecated
~~~~~~~~~~
* deprecate support to specify processor in form of a boolean (will be removed in v3.0.0)

  * new functions will not get support for this in the first place

* deprecate ``rapidfuzz.string_metric`` (will be removed in v3.0.0). Similar scorers are available
  in ``rapidfuzz.distance.*``

Fixed
~~~~~
* process.cdist did raise an exception when used with a pure python scorer

Performance
~~~~~~~~~~~
* improve performance and memory usage of ``rapidfuzz.string_metric.levenshtein_editops``

  * memory usage is reduced by 33%
  * performance is improved by around 10%-20%

* significantly improve performance of  ``rapidfuzz.string_metric.levenshtein`` for ``max <= 31``
  using a banded implementation

[1.9.1] - 2021-12-13
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix bug in new editops implementation, causing it to SegFault on some inputs (see qurator-spk/dinglehopper#64)

[1.9.0] - 2021-12-11
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix some issues in the type annotations (see #163)

Performance
~~~~~~~~~~~
* improve performance and memory usage of ``rapidfuzz.string_metric.levenshtein_editops``

  * memory usage is reduced by 10x
  * performance is improved from ``O(N * M)`` to ``O([N / 64] * M)``

[1.8.3] - 2021-11-19
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* Added missing wheels for Python3.6 on MacOs and Windows (see #159)

[1.8.2] - 2021-10-27
^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* Add wheels for Python 3.10 on MacOs

[1.8.1] - 2021-10-22
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix incorrect editops results (See #148)

[1.8.0] - 2021-10-20
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* Add Wheels for Python3.10 on all platforms except MacOs (see #141)
* Improve performance of ``string_metric.jaro_similarity`` and  ``string_metric.jaro_winkler_similarity`` for strings with a length <= 64

[1.7.1] - 2021-10-02
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fixed incorrect results of fuzz.partial_ratio for long needles (see #138)

[1.7.0] - 2021-09-27
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* Added typing for process.cdist
* Added multithreading support to cdist using the argument ``process.cdist``
* Add dtype argument to ``process.cdist`` to set the dtype of the result numpy array (see #132)
* Use a better hash collision strategy in the internal hashmap, which improves the worst case performance

[1.6.2] - 2021-09-15
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* improved performance of fuzz.ratio
* only import process.cdist when numpy is available

[1.6.1] - 2021-09-11
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* Add back wheels for Python2.7

[1.6.0] - 2021-09-10
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* fuzz.partial_ratio uses a new implementation for short needles (<= 64). This implementation is

  * more accurate than the current implementation (it is guaranteed to find the optimal alignment)
  * it is significantly faster

* Add process.cdist to compare all elements of two lists (see #51)

[1.5.1] - 2021-09-01
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix out of bounds access in levenshtein_editops

[1.5.0] - 2021-08-21
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* all scorers do now support similarity/distance calculations between any sequence of hashables. So it is possible to calculate e.g. the WER as:
  .. code-block::

     >>> string_metric.levenshtein(["word1", "word2"], ["word1", "word3"])
     1

Added
~~~~~
* Added type stub files for all functions
* added jaro similarity in ``string_metric.jaro_similarity``
* added jaro winkler similarity in ``string_metric.jaro_winkler_similarity``
* added Levenshtein editops in ``string_metric.levenshtein_editops``

Fixed
~~~~~
* Fixed support for set objects in ``process.extract``
* Fixed inconsistent handling of empty strings

[1.4.1] - 2021-03-30
^^^^^^^^^^^^^^^^^^^^
Performance
~~~~~~~~~~~
* improved performance of result creation in process.extract

Fixed
~~~~~
* Cython ABI stability issue (#95)
* fix missing decref in case of exceptions in process.extract

[1.4.0] - 2021-03-29
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* added processor support to ``levenshtein`` and ``hamming``
* added distance support to extract/extractOne/extract_iter

Fixed
~~~~~
* incorrect results of ``normalized_hamming`` and ``normalized_levenshtein`` when used with ``utils.default_process`` as processor

[1.3.3] - 2021-03-20
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix a bug in the mbleven implementation of the uniform Levenshtein distance and cover it with fuzz tests

[1.3.2] - 2021-03-20
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* some of the newly activated warnings caused build failures in the conda-forge build

[1.3.1] - 2021-03-20
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fixed issue in LCS calculation for partial_ratio (see #90)
* Fixed incorrect results for normalized_hamming and normalized_levenshtein when the processor ``utils.default_process`` is used
* Fix many compiler warnings

[1.3.0] - 2021-03-16
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* add wheels for a lot of new platforms
* drop support for Python 2.7

Performance
~~~~~~~~~~~
* use ``is`` instead of ``==`` to compare functions directly by address

Fixed
~~~~~
* Fix another ref counting issue
* Fix some issues in the Levenshtein distance algorithm (see #92)

[1.2.1] - 2021-03-08
^^^^^^^^^^^^^^^^^^^^
Performance
~~~~~~~~~~~
* further improve bitparallel implementation of uniform Levenshtein distance for strings with a length > 64 (in many cases more than 50% faster)

[1.2.0] - 2021-03-07
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* add more benchmarks to documentation

Performance
~~~~~~~~~~~
* add bitparallel implementation to InDel Distance (Levenshtein with the weights 1,1,2) for strings with a length > 64
* improve bitparallel implementation of uniform Levenshtein distance for strings with a length > 64
* use the InDel Distance and uniform Levenshtein distance in more cases instead of the generic implementation
* Directly use the Levenshtein implementation in C++ instead of using it through Python in process.*

[1.1.2] - 2021-03-03
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix reference counting in process.extract (see #81)

[1.1.1] - 2021-02-23
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Fix result conversion in process.extract (see #79)

[1.1.0] - 2021-02-21
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* string_metric.normalized_levenshtein supports now all weights
* when different weights are used for Insertion and Deletion the strings are not swapped inside the Levenshtein implementation anymore. So different weights for Insertion and Deletion are now supported.
* replace C++ implementation with a Cython implementation. This has the following advantages:

  * The implementation is less error prone, since a lot of the complex things are done by Cython
  * slightly faster than the current implementation (up to 10% for some parts)
  * about 33% smaller binary size
  * reduced compile time

* Added \*\*kwargs argument to process.extract/extractOne/extract_iter that is passed to the scorer
* Add max argument to hamming distance
* Add support for whole Unicode range to utils.default_process

Performance
~~~~~~~~~~~
* replaced Wagner Fischer usage in the normal Levenshtein distance with a bitparallel implementation

[1.0.2] - 2021-02-19
^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* The bitparallel LCS algorithm in fuzz.partial_ratio did not find the longest common substring properly in some cases.
  The old algorithm is used again until this bug is fixed.

[1.0.1] - 2021-02-17
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* string_metric.normalized_levenshtein supports now the weights (1, 1, N) with N >= 1

Performance
~~~~~~~~~~~
* The Levenshtein distance with the weights (1, 1, >2) do now use the same implementation as the weight (1, 1, 2), since
  ``Substitution > Insertion + Deletion`` has no effect

Fixed
~~~~~
* fix uninitialized variable in bitparallel Levenshtein distance with the weight (1, 1, 1)

[1.0.0] - 2021-02-12
^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* all normalized string_metrics can now be used as scorer for process.extract/extractOne
* Implementation of the C++ Wrapper completely refactored to make it easier to add more scorers, processors and string matching algorithms in the future.
* increased test coverage, that already helped to fix some bugs and help to prevent regressions in the future
* improved docstrings of functions

Performance
~~~~~~~~~~~
* Added bit-parallel implementation of the Levenshtein distance for the weights (1,1,1) and (1,1,2).
* Added specialized implementation of the Levenshtein distance for cases with a small maximum edit distance, that is even faster, than the bit-parallel implementation.
* Improved performance of ``fuzz.partial_ratio``
  -> Since ``fuzz.ratio`` and ``fuzz.partial_ratio`` are used in most scorers, this improves the overall performance.
* Improved performance of ``process.extract`` and ``process.extractOne``

Deprecated
~~~~~~~~~~
* the ``rapidfuzz.levenshtein`` module is now deprecated and will be removed in v2.0.0
  These functions are now placed in ``rapidfuzz.string_metric``. ``distance``\ , ``normalized_distance``\ , ``weighted_distance`` and ``weighted_normalized_distance`` are combined into ``levenshtein`` and ``normalized_levenshtein``.

Added
~~~~~
* added normalized version of the hamming distance in ``string_metric.normalized_hamming``
* process.extract_iter as a generator, that yields the similarity of all elements, that have a similarity >= score_cutoff

Fixed
~~~~~
* multiple bugs in extractOne when used with a scorer, that's not from RapidFuzz
* fixed bug in ``token_ratio``
* fixed bug in result normalization causing zero division

[0.14.2] - 2020-12-31
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* utf8 usage in the copyright header caused problems with python2.7 on some platforms (see #70)

[0.14.1] - 2020-12-13
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* when a custom processor like ``lambda s: s`` was used with any of the methods inside fuzz.* it always returned a score of 100. This release fixes this and adds a better test coverage to prevent this bug in the future.

[0.14.0] - 2020-12-09
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added hamming distance metric in the levenshtein module

Performance
~~~~~~~~~~~
* improved performance of default_process by using lookup table

[0.13.4] - 2020-11-30
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Add missing virtual destructor that caused a segmentation fault on Mac Os

[0.13.3] - 2020-11-21
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* C++11 Support
* manylinux wheels

[0.13.2] - 2020-11-21
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* Levenshtein was not imported from __init__
* The reference count of a Python Object inside process.extractOne was decremented to early

[0.13.1] - 2020-11-17
^^^^^^^^^^^^^^^^^^^^^
Performance
~~~~~~~~~~~
* process.extractOne  exits early when a score of 100 is found. This way the other strings do not have to be preprocessed anymore.

[0.13.0] - 2020-11-16
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* string objects passed to scorers had to be strings even before preprocessing them. This was changed, so they only have to be strings after preprocessing similar to process.extract/process.extractOne

Performance
~~~~~~~~~~~
* process.extractOne is now implemented in C++ making it a lot faster
* When token_sort_ratio or partial_token_sort ratio is used inprocess.extractOne the words in the query are only sorted once to improve the runtime

Changed
~~~~~~~
* process.extractOne/process.extract do now return the index of the match, when the choices are a list.

Removed
~~~~~~~
* process.extractIndices got removed, since the indices are now already returned by process.extractOne/process.extract

[0.12.5] - 2020-10-26
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix documentation of process.extractOne (see #48)

[0.12.4] - 2020-10-22
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* Added wheels for

  * CPython 2.7 on windows 64 bit
  * CPython 2.7 on windows 32 bit
  * PyPy 2.7 on windows 32 bit

[0.12.3] - 2020-10-09
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix bug in partial_ratio (see #43)

[0.12.2] - 2020-10-01
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix inconsistency with fuzzywuzzy in partial_ratio when using strings of equal length

[0.12.1] - 2020-09-30
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* MSVC has a bug and therefore crashed on some of the templates used. This Release simplifies the templates so compiling on msvc works again

[0.12.0] - 2020-09-30
^^^^^^^^^^^^^^^^^^^^^
Performance
~~~~~~~~~~~
* partial_ratio is using the Levenshtein distance now, which is a lot faster. Since many of the other algorithms use partial_ratio, this helps to improve the overall performance

[0.11.3] - 2020-09-22
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* fix partial_token_set_ratio returning 100 all the time

[0.11.2] - 2020-09-12
^^^^^^^^^^^^^^^^^^^^^
Added
~~~~~
* added rapidfuzz.__author__, rapidfuzz.__license__ and rapidfuzz.__version__

[0.11.1] - 2020-09-01
^^^^^^^^^^^^^^^^^^^^^
Fixed
~~~~~
* do not use auto junk when searching the optimal alignment for partial_ratio

[0.11.0] - 2020-08-22
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* support for python 2.7 added #40
* add wheels for python2.7 (both pypy and cpython) on MacOS and Linux

[0.10.0] - 2020-08-17
^^^^^^^^^^^^^^^^^^^^^
Changed
~~~~~~~
* added wheels for Python3.9

Fixed
~~~~~
* tuple scores in process.extractOne are now supported #39
