#pragma once
#include <string_view>
#include <vector>
#include <cmath>
#include "concepts.hpp"
#include "utils.hpp"


namespace levenshtein {
    enum EditType {
      EditKeep,
      EditReplace,
      EditInsert,
      EditDelete,
    };

    struct EditOp {
      EditType op_type;
      size_t first_start;
      size_t second_start;
    };

    struct Matrix {
        size_t prefix_len;
        std::vector<size_t> matrix;
        size_t matrix_columns;
        size_t matrix_rows;
    };

    Matrix matrix(std::string_view sentence1, std::string_view sentence2);

    std::vector<EditOp> editops(std::string_view sentence1, std::string_view sentence2);




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
    size_t weighted_distance(std::string_view sentence1, std::string_view sentence2,
                             std::string_view delimiter="");
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
                             size_t max_distance, std::string_view delimiter="");
    size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                             size_t max_distance, std::string_view delimiter="");

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    template<Iterable Sentence1, Iterable Sentence2>
    float normalized_weighted_distance(Sentence1 sentence1, Sentence2 sentence2,
                                       float min_ratio=0.0, std::string_view delimiter="")
    {
        if (sentence1.empty() && sentence2.empty()) {
            return 1.0;
        }

        size_t lensum = recursiveIterableSize(sentence1, delimiter.size()) + recursiveIterableSize(sentence2, delimiter.size());
        if (!min_ratio) {
            return 1.0 - (float)weighted_distance(sentence1, sentence2, delimiter) / (float)lensum;
        }

        size_t max_distance = static_cast<size_t>(std::ceil((float)lensum - min_ratio * lensum));
        size_t distance = weighted_distance(sentence1, sentence2, max_distance, delimiter);
        if (distance == std::numeric_limits<size_t>::max()) {
            return 0.0;
        }
        return 1.0 - (float)distance / (float)lensum;
    }
}
