#pragma once
#include <string_view>
#include <vector>


namespace levenshtein {
    //Not implemented yet

    /**
     * Calculates the minimum number of insertions, deletions, and substitutions
     * required to change one sequence into the other according to Levenshtein.
     * Each edit operation has a similar cost of 1.
     */
    size_t distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    size_t distance(std::string_view sentence1, std::string_view sentence2);

    /**
    * Calculates a normalized score of the Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    size_t normalized_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    size_t normalized_distance(std::string_view sentence1, std::string_view sentence2);


    // should be more generic so at least any combination of string view and vector of string view can be used
    // should be even possible to use for other types like a vector of integers
    // the delimiter like e.g. ' ' for a vector of string views should be specified by the user of the function
    // should be possible to use any iterable/single value. e.g. when using a string it could be any iterable that returns chars aswell like e.g. a string_view
    // or with a vector of integers could be both a iterable of integers or a single integer
    // iterable of iterable is probably the maximum that is needed for now since it allows using e.g. a vector of string views

    /**
     * Calculates the minimum number of insertions, deletions, and substitutions
     * required to change one sequence into the other according to Levenshtein.
     * Opposed to the normal distance function which has a cost of 1 for all edit operations,
     * it uses the following costs for edit operations:
     *
     * edit operation | cost
     * :------------- | :---
     * Insert         | 1
     * Remove         | 1
     * Replace        | 2
     */
    size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    size_t weighted_distance(std::string_view sentence1, std::string_view sentence2);

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    size_t normalized_weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    size_t normalized_weighted_distance(std::string_view sentence1, std::string_view sentence2);
}
