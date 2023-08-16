# API differences to `fuzzywuzzy`

`Rapidfuzz` does provide a very similar API to `fuzzywuzzy`/`thefuzz` making it a drop in replacement for a large amount of projects.
However there are some differences which are listed below:

## ratio implementation

`fuzzywuzzy` provides two implementations of the algorithm:
1) a pure Python version implemented using difflib (Ratcliff and Obershelp algorithm)
2) an accelerated version using the Indel similarity (similar to the Levenshtein distance but only allows for Insertions / Deletions)

This leads to different results depending on the version in use. `RapidFuzz` always uses the Indel similarity both in the pure Python
fallback implementation and the C++ based implementation to provide consistent matching results.

## partial_ratio implementation
`fuzzywuzzy` searches for the optimal matching substring and then calculates the similarity using `ratio`. This substring is searches using either:
1) `difflib.SequenceMatcher.get_matching_blocks` (based on Ratcliff and Obershelp algorithm)
2) `Levenshtein.matching_blocks` (backtracks Levenshtein matrix)

This implementation has a couple of issues:
1) in the pure Python implementation the automatic junk heuristic of difflib is not deactivated. This heuristic improves the performance for long strings,
but can lead to completely incorrect results.
2) the accelerated version backtracks the Levenshtein matrix to find the same alignment found by the Python implementation. However the algorithm just uses
one of multiple optimal alignment. There is no guarantee for this alignment to include the longest common substring.
3) the optimal substring is assumed to start at one of these `matching_blocks`. However this is not guaranteed.

`RapidFuzz` uses a sliding window approach (with some optimizations to skip impossible alignments) to find the optimal alignment. This approach is guaranteed
to find the optimal alignment.

## differences in preprocessing

`fuzzywuzzy` provides the function `utils.full_process` to preprocess strings. This function is called `utils.default_process` in `RapidFuzz`. It behaves similar with the only exception
that it does not provide the optional argument `force_ascii` which removes any non ascii characters from a string.

## differences in scorers

`fuzzywuzzy` has the following scorers which preprocess strings by default:
- `fuzz.token_sort_ratio`
- `fuzz.token_set_ratio`
- `fuzz.partial_token_sort_ratio`
- `fuzz.partial_token_set_ratio`
- `fuzz.WRatio`
- `fuzz.QRatio`
- `fuzz.UWRatio`
- `fuzz.UQRatio`

With the exception `fuzz.UWRatio` and `fuzz.UQRatio` of all have `force_ascii` enabled forthe peprocessing function by default.

In `RapidFuzz` no scorer preprocesses strings by default to keep the interface consistent. However a preprocessing function can be provided using the `processor` argument. In addition the functions `fuzz.UWRatio` and `fuzz.UQRatio` do not exist, since they are the same as  `fuzz.WRatio` / `fuzz.QRatio` with `force_ascii` disabled. Since in `RapidFuzz` the `force_ascii` argument does not exist these functions do not provide any value.

## differences in processor functions

In `fuzzywuzzy` the process module includes the following functions:
- `extractWithoutOrder` (generator over unsorted results)
- `extract` (find the N best matches in a sorted list)
- `extractBests` (same as extract but with an addition score_cutoff parameter to filter bad matches)
- `extractOne` (find best match)
- `dedupe` (deduplicate list)

In `RapidFuzz` these functions are sometimes available under different names:
- `extractWithoutOrder` is called `extract_iter`
- `extract` / `extractBests` are a single function called `extract` which povides the optional `score_cutoff` argument
- `extractOne` is available under the same name
- `dedupe` is not available

In addition these functions do not preprocess strings by default. However preprocessing can be enabled using the `processor` argument.
