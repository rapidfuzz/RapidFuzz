#pragma once
#include <string_view>
#include <vector>
#include <cmath>
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
      std::size_t first_start;
      std::size_t second_start;
      EditOp(EditType op_type, std::size_t first_start, std::size_t second_start)
        : op_type(op_type), first_start(first_start), second_start(second_start) {}
    };

    struct Matrix {
        std::size_t prefix_len;
        std::vector<std::size_t> matrix;
        std::size_t matrix_columns;
        std::size_t matrix_rows;
    };

    Matrix matrix(std::string_view sentence1, std::string_view sentence2);

    std::vector<EditOp> editops(std::string_view sentence1, std::string_view sentence2);

    struct MatchingBlock {
    	std::size_t first_start;
    	std::size_t second_start;
    	std::size_t len;
      MatchingBlock(std::size_t first_start, std::size_t second_start, std::size_t len)
        : first_start(first_start), second_start(second_start), len(len) {}
    };

    std::vector<MatchingBlock> matching_blocks(std::string_view sentence1, std::string_view sentence2);


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
    std::size_t weighted_distance(std::string_view sentence1, std::string_view sentence2,
                             std::string_view delimiter="");
    std::size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                             std::string_view delimiter="");


    /**
     * These functions allow providing a max_distance parameter that can be used to exit early when the
     * calculated levenshtein distance is at least as big as max_distance and will return the maximal
     * possible value for std::size_t.
     * This range check makes the levenshtein calculation about 20% slower, so it should be only used
     * when it can usually exit early.
     */
    std::size_t weighted_distance(std::string_view sentence1, std::string_view sentence2,
                             std::size_t max_distance, std::string_view delimiter="");
    std::size_t weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                             std::size_t max_distance, std::string_view delimiter="");

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    template<typename Sentence1, typename Sentence2>
    float normalized_weighted_distance(const Sentence1 &sentence1, const Sentence2 &sentence2,
                                       float min_ratio=0.0, std::string_view delimiter="")
    {
        if (sentence1.empty() && sentence2.empty()) {
            return 1.0;
        }

        if (sentence1.empty() || sentence1.empty()) {
          return 0.0;
        }

        std::size_t sentence1_len = recursiveIterableSize(sentence1, delimiter.size());
        std::size_t sentence2_len = recursiveIterableSize(sentence2, delimiter.size());
        std::size_t lensum = sentence1_len + sentence2_len;

        // constant time calculation to find a string ratio based on the string length
        // so it can exit early without running any levenshtein calculations
        std::size_t min_distance = (sentence1_len > sentence2_len)
          ? sentence1_len - sentence2_len
          : sentence2_len - sentence1_len;

        float len_ratio = 1.0 - (float)min_distance / (float)lensum;
        if (len_ratio < min_ratio) {
          return 0.0;
        }

        // TODO: this needs more thoughts when to start using score cutoff, since it performs slower when it can not exit early
        // -> just because it has a smaller ratio does not mean levenshtein can always exit early
        // has to be tested with some more real examples
        std::size_t distance = (min_ratio > 0.7)
          ? weighted_distance(sentence1, sentence2, std::ceil((float)lensum - min_ratio * lensum), delimiter)
          : weighted_distance(sentence1, sentence2, delimiter);

        if (distance == std::numeric_limits<std::size_t>::max()) {
            return 0.0;
        }
        return 1.0 - (float)distance / (float)lensum;
    }
}
