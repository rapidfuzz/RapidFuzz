#pragma once
#include <boost/utility/string_view.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include "utils.hpp"

namespace levenshtein {
struct WeightTable {
    std::size_t insert_cost;
    std::size_t delete_cost;
    std::size_t replace_cost;
};

enum EditType {
    EditKeep,
    EditReplace,
    EditInsert,
    EditDelete,
};

struct Matrix {
    std::size_t prefix_len;
    std::vector<std::size_t> matrix;
    std::size_t matrix_columns;
    std::size_t matrix_rows;
};

struct MatchingBlock {
    std::size_t first_start;
    std::size_t second_start;
    std::size_t len;
    MatchingBlock(std::size_t first_start, std::size_t second_start, std::size_t len)
        : first_start(first_start)
        , second_start(second_start)
        , len(len)
    {}
};

struct EditOp {
    EditType op_type;
    std::size_t first_start;
    std::size_t second_start;
    EditOp(EditType op_type, std::size_t first_start, std::size_t second_start)
        : op_type(op_type)
        , first_start(first_start)
        , second_start(second_start)
    {}
};


template<typename CharT>
Matrix matrix(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2);

template <typename CharT>
Matrix matrix(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2);

template<typename CharT>
std::vector<MatchingBlock> matching_blocks(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2);

template <typename CharT>
std::vector<MatchingBlock> matching_blocks(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2);

template<typename CharT>
double normalized_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    double min_ratio = 0.0);

template <typename CharT>
double normalized_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    double min_ratio = 0.0);

template<typename CharT>
std::size_t distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2);

template <typename CharT>
std::size_t distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2);


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
 * 
 * @param sentence1 first sentence to match (can be either a string type or a vector of strings)
 * @param sentence2 second sentence to match (can be either a string type or a vector of strings)
 * @return weighted levenshtein distance
 */
template<typename CharT>
std::size_t weighted_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2);

template <typename CharT>
std::size_t weighted_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2);

template<typename CharT>
std::size_t generic_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    WeightTable weights = { 1, 1, 1 });

template <typename CharT>
std::size_t generic_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    WeightTable weights = { 1, 1, 1 });

/**
  * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
  * 1.0 (inclusive), where 1.0 means the sequences are the same.
  */
template<typename CharT>
double normalized_weighted_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    double min_ratio = 0.0);

template <typename CharT>
double normalized_weighted_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    double min_ratio = 0.0);

}
#include "levenshtein.txx"
