#pragma once
#include <string_view>
#include <vector>
#include <cmath>
#include <optional>
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

struct Matrix {
    std::size_t prefix_len;
    std::vector<std::size_t> matrix;
    std::size_t matrix_columns;
    std::size_t matrix_rows;
};

Matrix matrix(std::wstring_view sentence1, std::wstring_view sentence2);

std::vector<EditOp> editops(std::wstring_view sentence1, std::wstring_view sentence2);

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

std::vector<MatchingBlock> matching_blocks(std::wstring_view sentence1, std::wstring_view sentence2);

double normalized_distance(std::wstring_view sentence1, std::wstring_view sentence2, double min_ratio = 0.0);

std::size_t distance(std::wstring_view sentence1, std::wstring_view sentence2);

template <typename MaxDistanceCalc = std::false_type>
auto levenshtein_word_cmp(const wchar_t& letter_cmp, const std::vector<std::wstring_view>& words,
    std::vector<std::size_t>& cache, std::size_t current_cache);

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
 * @param max_distance maximum distance to exit early. When using this the calculation is about 20% slower
 *                     so when it can not exit early it should not be used
 * @return weighted levenshtein distance
 */
template <typename MaxDistance = std::nullopt_t>
std::size_t weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, MaxDistance max_distance = std::nullopt);

template <typename MaxDistance = std::nullopt_t>
std::size_t weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2, MaxDistance max_distance = std::nullopt);

std::size_t generic_distance(std::wstring_view source, std::wstring_view target, WeightTable weights = { 1, 1, 1 });

/**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
template <typename Sentence1, typename Sentence2>
double normalized_weighted_distance(const Sentence1& sentence1, const Sentence2& sentence2, double min_ratio = 0.0);
}

template <typename MaxDistanceCalc>
inline auto levenshtein::levenshtein_word_cmp(const wchar_t& letter_cmp, const std::vector<std::wstring_view>& words,
    std::vector<std::size_t>& cache, std::size_t current_cache)
{
    std::size_t result = current_cache + 1;
    auto cache_iter = cache.begin();
    auto word_iter = words.begin();
    auto min_distance = std::numeric_limits<std::size_t>::max();

    auto charCmp = [&](const wchar_t& char2) {
        if (letter_cmp == char2) {
            result = current_cache;
        } else {
            ++result;
        }

        current_cache = *cache_iter;
        if (result > current_cache + 1) {
            result = current_cache + 1;
        }

        if constexpr(!std::is_same_v<std::false_type, MaxDistanceCalc>) {
            if (current_cache < min_distance) {
                min_distance = current_cache;
            }
        }

        *cache_iter = result;
        ++cache_iter;
    };

    // no whitespace should be added in front of the first word
    for (const auto& letter : *word_iter) {
        charCmp(letter);
    }
    ++word_iter;

    for (; word_iter != words.end(); ++word_iter) {
        // between every word there should be one whitespace
        charCmp(0x20);

        // check following word
        for (const auto& letter : *word_iter) {
            charCmp(letter);
        }
    }

    if constexpr(!std::is_same_v<std::false_type, MaxDistanceCalc>) {
        return min_distance;
    }
}

template <typename MaxDistance>
inline std::size_t levenshtein::weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2, MaxDistance max_distance)
{
    utils::remove_common_affix(sentence1, sentence2);
    std::size_t sentence1_len = utils::joined_size(sentence1);
    std::size_t sentence2_len = utils::joined_size(sentence2);

    if (sentence2_len > sentence1_len) {
        std::swap(sentence1, sentence2);
        std::swap(sentence1_len, sentence2_len);
    }

    if (!sentence2_len) {
        return sentence1_len;
    }

    std::vector<std::size_t> cache(sentence2_len);
    std::iota(cache.begin(), cache.end(), 1);

    std::size_t range1_pos = 0;
    auto word_iter = sentence1.begin();

    // no delimiter in front of first word
    for (const auto& letter : *word_iter) {
        if constexpr(!std::is_same_v<MaxDistance, std::nullopt_t>) {
            std::size_t min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos);
            if (min_distance > max_distance) {
                return std::numeric_limits<std::size_t>::max();
            }
        } else {
            levenshtein_word_cmp(letter, sentence2, cache, range1_pos);
        }

        ++range1_pos;
    }

    ++word_iter;
    for (; word_iter != sentence1.end(); ++word_iter) {
        // whitespace between words
        if constexpr(!std::is_same_v<MaxDistance, std::nullopt_t>) {
            std::size_t min_distance = levenshtein_word_cmp<std::true_type>(static_cast<wchar_t>(0x20), sentence2, cache, range1_pos);
            if (min_distance > max_distance) {
                return std::numeric_limits<std::size_t>::max();
            }
        } else {
            levenshtein_word_cmp(static_cast<wchar_t>(0x20), sentence2, cache, range1_pos);
        }

        ++range1_pos;

        for (const auto& letter : *word_iter) {
            if constexpr(!std::is_same_v<MaxDistance, std::nullopt_t>) {
                std::size_t min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos);
                if (min_distance > max_distance) {
                    return std::numeric_limits<std::size_t>::max();
                }
            } else {
                levenshtein_word_cmp(letter, sentence2, cache, range1_pos);
            }

            ++range1_pos;
        }
    }

    return cache.back();
}

template <typename MaxDistance>
inline std::size_t levenshtein::weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, MaxDistance max_distance)
{
    utils::remove_common_affix(sentence1, sentence2);

    if (sentence2.size() > sentence1.size()) {
        std::swap(sentence1, sentence2);
    }

    if (sentence2.empty()) {
        return sentence1.length();
    }

    std::vector<std::size_t> cache(sentence2.length());
    std::iota(cache.begin(), cache.end(), 1);

    std::size_t sentence1_pos = 0;
    for (const auto& char1 : sentence1) {
        auto cache_iter = cache.begin();
        std::size_t current_cache = sentence1_pos;
        std::size_t result = sentence1_pos + 1;
        auto min_distance = std::numeric_limits<std::size_t>::max();
        for (const auto& char2 : sentence2) {
            if (char1 == char2) {
                result = current_cache;
            } else {
                ++result;
            }
            current_cache = *cache_iter;
            if (result > current_cache + 1) {
                result = current_cache + 1;
            }

            // only check max distance when one is supplied
            if constexpr(!std::is_same_v<MaxDistance, std::nullopt_t>) {
                if (current_cache < min_distance) {
                        min_distance = current_cache;
                }
            }
            *cache_iter = result;
            ++cache_iter;
        }

        // only check max distance when one is supplied
        if constexpr(!std::is_same_v<MaxDistance, std::nullopt_t>) {
            if (min_distance > max_distance) {
                return std::numeric_limits<std::size_t>::max();
            }
        }
        ++sentence1_pos;
    }
    return cache.back();
}

template <typename Sentence1, typename Sentence2>
inline double levenshtein::normalized_weighted_distance(const Sentence1& sentence1, const Sentence2& sentence2, double min_ratio)
{
    if (sentence1.empty() || sentence2.empty()) {
        return sentence1.empty() && sentence2.empty();
    }

    std::size_t sentence1_len = utils::joined_size(sentence1);
    std::size_t sentence2_len = utils::joined_size(sentence2);
    std::size_t lensum = sentence1_len + sentence2_len;

    // constant time calculation to find a string ratio based on the string length
    // so it can exit early without running any levenshtein calculations
    std::size_t min_distance = (sentence1_len > sentence2_len)
        ? sentence1_len - sentence2_len
        : sentence2_len - sentence1_len;

    double len_ratio = 1.0 - static_cast<double>(min_distance) / lensum;
    if (len_ratio < min_ratio) {
        return 0.0;
    }

    // TODO: this needs more thoughts when to start using score cutoff, since it performs slower when it can not exit early
    // -> just because it has a smaller ratio does not mean levenshtein can always exit early
    // has to be tested with some more real examplesstatic_cast<double>(
    std::size_t dist = (min_ratio > 0.7)
        ? weighted_distance(sentence1, sentence2, std::ceil(static_cast<double>(lensum) - min_ratio * lensum))
        : weighted_distance(sentence1, sentence2);

    if (dist > lensum) {
        return 0.0;
    }
    double ratio = 1.0 - static_cast<double>(dist) / lensum;
    return (ratio >= min_ratio) ? ratio : 0.0;
}
