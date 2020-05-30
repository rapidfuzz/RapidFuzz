---
template: overrides/main.html
---

# fuzz

All string matching algorithms in fuzz share a common interface to simplify the usage.
However the methods are sometimes using differing default values to make the behaviour similar to the behaviour of FuzzyWuzzy.

=== "Python"

    Parameters:

    - **s1**: *str*

        First string to compare.

    - **s2**: *str*

        Second string to compare.

    - **processor**: *(Union[bool, Callable])*

        Optional callable that reformats the strings.

        - `None` is used by default in `fuzz.ratio` and `fuzz.partial_ratio`.
        - `utils.default_process`, which lowercases the strings, trims whitespaces and removes non alphanumeric characters, is used by default for all other `fuzz.*ratio` methods

    - **score_cutoff**: *float*, default `0`

        Optional argument for a score threshold as a float between 0 and 100. For `ratio < score_cutoff`, 0 is returned instead.

    Returns:

    - **score**: *float*

        Ratio between `s1` and `s2` as a float between 0 and 100

=== "C++"


## ratio

Calculates a simple ratio between two strings.

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.

=== "Python"
    ```bash
    > from rapidfuzz import fuzz
    > fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    ```

=== "C++"
    ```cpp
    #include "fuzz.hpp"
    using rapidfuzz::fuzz::ratio;

    // score is 96.55171966552734
    double score = rapidfuzz::fuzz::ratio("this is a test", "this is a test!");
    ```


## partial_ratio

Calculates the [ratio](#ratio) of the optimal string alignment

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.

=== "Python"
    ```bash
    > from rapidfuzz import fuzz
    > fuzz.partial_ratio("this is a test", "this is a test!")
    100
    ```

=== "C++"
    ```cpp
    #include "fuzz.hpp"
    using rapidfuzz::fuzz::partial_ratio;

    // score is 100
    double score = rapidfuzz::fuzz::partial_ratio("this is a test", "this is a test!");
    ```

## token_sort_ratio

Sorts the words in the strings and calculates the [ratio](#ratio) between them.

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.

=== "Python"
    ```bash
    > from rapidfuzz import fuzz
    > fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100
    ```

=== "C++"
    ```cpp
    #include "fuzz.hpp"
    using rapidfuzz::fuzz::token_sort_ratio;

    // score is 100
    double score = token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    ```

## partial_token_sort_ratio

Sorts the words in the strings and calculates the [partial_ratio](#partial_ratio) between them.

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.


## token_set_ratio

Compares the words in the strings based on unique and common words between them using [ratio](#ratio).

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.

=== "Python"
    ```bash
    > fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    83.8709716796875
    > fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100.0
    ```

=== "C++"
    ```cpp
    #include "fuzz.hpp"
    using rapidfuzz::fuzz::token_sort_ratio;
	using rapidfuzz::fuzz::token_set_ratio;

    // score1 is 83.87
    double score1 = token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    // score2 is 100
    double score2 = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    ```

## partial_token_set_ratio

Compares the words in the strings based on unique and common words between them using [partial_ratio](#partial_ratio).

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details..


## token_ratio

Helper method that returns the maximum of [token_set_ratio](#token_set_ratio) and
[token_sort_ratio](#token_sort_ratio) (faster than manually executing the two functions)

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.


## partial_token_ratio

Helper method that returns the maximum of [partial_token_set_ratio](#partial_token_set_ratio) and
[partial_token_sort_ratio](#partial_token_sort_ratio) (faster than manually executing the two functions)

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.


## QRatio
Similar algorithm to [ratio](#ratio), but preprocesses the strings by default, while it does not do this by default in
[ratio](#ratio).

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.

## WRatio
Calculates a weighted ratio based on the other ratio algorithms.

Parameters: Check [fuzz](#fuzz) for further details.

Returns: Check [fuzz](#fuzz) for further details.
