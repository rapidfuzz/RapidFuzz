#pragma once
#include <string_view>
#include <vector>


namespace levenshtein {
    struct Matrix {
        size_t prefix_len;
        std::vector<size_t> matrix;
        size_t matrix_columns;
        size_t matrix_rows;
    };

    Matrix matrix(std::string_view sentence1, std::string_view sentence2);


    /* Not implemented yet */

    /**
     * Calculates the minimum number of insertions, deletions, and substitutions
     * required to change one sequence into the other according to Levenshtein.
     * Each edit operation has a similar cost of 1.
     */
    //size_t distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    //size_t distance(std::string_view sentence1, std::string_view sentence2);

    /**
    * Calculates a normalized score of the Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    //float normalized_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2);
    //float normalized_distance(std::string_view sentence1, std::string_view sentence2);


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
    size_t weighted_distance(std::string_view sentence1, std::string_view sentence2);
    size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                             std::string_view delimiter="");


    /**
     * These functions allow providing a max_distance parameter that can be used to exit early when the
     * calculated levenshtein distance is at least as big as max_distance and will return the maximal
     * possible value for size_t.
     * This range check makes the levenshtein calculation about 20% slower, so it should be only used
     * when it can usually exit early.
     */
    size_t weighted_distance(std::string_view sentence1, std::string_view sentence2,
                             size_t max_distance);
    size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                             size_t max_distance, std::string_view delimiter="");

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    float normalized_weighted_distance(std::string_view sentence1, std::string_view sentence2);
    float normalized_weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                                       std::string_view delimiter="");

    float normalized_weighted_distance(std::string_view sentence1, std::string_view sentence2,
                                       float min_ratio);
    float normalized_weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                                       float min_ratio, std::string_view delimiter="");
}
