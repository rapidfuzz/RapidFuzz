#include "levenshtein.hpp"
#include <algorithm>
#include <stdexcept>


template<typename CharT>
levenshtein::Matrix levenshtein::matrix(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2)
{
    Affix affix = utils::remove_common_affix(sentence1, sentence2);

    std::size_t matrix_columns = sentence1.length() + 1;
    std::size_t matrix_rows = sentence2.length() + 1;

    std::vector<std::size_t> cache_matrix(matrix_rows * matrix_columns, 0);

    for (std::size_t i = 0; i < matrix_rows; ++i) {
        cache_matrix[i] = i;
    }

    for (std::size_t i = 1; i < matrix_columns; ++i) {
        cache_matrix[matrix_rows * i] = i;
    }

    std::size_t sentence1_pos = 0;
    for (const auto& char1 : sentence1) {
        auto prev_cache = cache_matrix.begin() + sentence1_pos * matrix_rows;
        auto result_cache = cache_matrix.begin() + (sentence1_pos + 1) * matrix_rows + 1;
        std::size_t result = sentence1_pos + 1;
        for (const auto& char2 : sentence2) {
            result = std::min({ result + 1,
                *prev_cache + (char1 != char2),
                *(++prev_cache) + 1 });
            *result_cache = result;
            ++result_cache;
        }
        ++sentence1_pos;
    }

    return Matrix{
        affix.prefix_len,
        cache_matrix,
        matrix_columns,
        matrix_rows
    };
}

template <typename CharT>
levenshtein::Matrix levenshtein::matrix(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2)
{
    return matrix(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2));
}

template<typename CharT>
inline std::vector<levenshtein::EditOp> editops(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2)
{
    auto m =  levenshtein::matrix(sentence1, sentence2);
    std::size_t matrix_columns = m.matrix_columns;
    std::size_t matrix_rows = m.matrix_rows;
    std::size_t prefix_len = m.prefix_len;
    auto lev_matrix = m.matrix;

    std::vector<levenshtein::EditOp> ops;
    ops.reserve(lev_matrix[matrix_columns * matrix_rows - 1]);

    std::size_t i = matrix_columns - 1;
    std::size_t j = matrix_rows - 1;
    std::size_t position = matrix_columns * matrix_rows - 1;

    auto is_replace = [=](std::size_t pos) {
        return lev_matrix[pos - matrix_rows - 1] < lev_matrix[pos];
    };
    auto is_insert = [=](std::size_t pos) {
        return lev_matrix[pos - 1] < lev_matrix[pos];
    };
    auto is_delete = [=](std::size_t pos) {
        return lev_matrix[pos - matrix_rows] < lev_matrix[pos];
    };
    auto is_keep = [=](std::size_t pos) {
        return lev_matrix[pos - matrix_rows - 1] == lev_matrix[pos];
    };

    while (i > 0 || j > 0) {
         levenshtein::EditType op_type;

        if (i && j && is_replace(position)) {
            op_type =  levenshtein::EditType::EditReplace;
            --i;
            --j;
            position -= matrix_rows + 1;
        } else if (j && is_insert(position)) {
            op_type =  levenshtein::EditType::EditInsert;
            --j;
            --position;
        } else if (i && is_delete(position)) {
            op_type =  levenshtein::EditType::EditDelete;
            --i;
            position -= matrix_rows;
        } else if (is_keep(position)) {
            --i;
            --j;
            position -= matrix_rows + 1;
            // EditKeep does not has to be stored
            continue;
        } else {
            throw std::logic_error("something went wrong extracting the editops from the levenshtein matrix");
        }

        ops.emplace_back(op_type, i + prefix_len, j + prefix_len);
    }

    std::reverse(ops.begin(), ops.end());
    return ops;
}

template<typename CharT>
std::vector<levenshtein::MatchingBlock> levenshtein::matching_blocks(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2)
{
    auto edit_ops = editops(sentence1, sentence2);
    std::size_t first_start = 0;
    std::size_t second_start = 0;
    std::vector<MatchingBlock> mblocks;

    for (const auto& op : edit_ops) {
        if (op.op_type == EditType::EditKeep) {
            continue;
        }

        if (first_start < op.first_start || second_start < op.second_start) {
            mblocks.emplace_back(first_start, second_start, op.first_start - first_start);
            first_start = op.first_start;
            second_start = op.second_start;
        }

        switch (op.op_type) {
        case EditType::EditReplace:
            first_start += 1;
            second_start += 1;
            break;
        case EditType::EditDelete:
            first_start += 1;
            break;
        case EditType::EditInsert:
            second_start += 1;
            break;
        case EditType::EditKeep:
            break;
        }
    }

    mblocks.emplace_back(sentence1.length(), sentence2.length(), 0);
    return mblocks;
}

template <typename CharT>
std::vector<levenshtein::MatchingBlock> levenshtein::matching_blocks(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2)
{
    return matching_blocks(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2));
}

template<typename CharT>
std::size_t levenshtein::distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2)
{
    utils::remove_common_affix(sentence1, sentence2);

    if (sentence2.size() > sentence1.size())
        std::swap(sentence1, sentence2);

    if (sentence2.empty()) {
        return sentence1.length();
    }

    std::vector<std::size_t> cache(sentence2.length() + 1);
    std::iota(cache.begin(), cache.end(), 0);

    for (const auto& char1 : sentence1) {
        auto cache_iter = cache.begin();
        std::size_t temp = *cache_iter;
        *cache_iter += 1;

        for (const auto& char2 : sentence2) {
            if (char1 != char2) {
                ++temp;
            }

            temp = std::min({ *cache_iter + 1,
                *(++cache_iter) + 1,
                temp });
            std::swap(*cache_iter, temp);
        }
    }
    return cache.back();
}

template <typename CharT>
std::size_t levenshtein::distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2)
{
    return distance(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2));
}

template<typename CharT>
std::size_t levenshtein::weighted_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2)
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

            *cache_iter = result;
            ++cache_iter;
        }

        ++sentence1_pos;
    }
    return cache.back();
}

template <typename CharT>
std::size_t levenshtein::weighted_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2)
{
    return weighted_distance(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2));
}

template<typename CharT>
std::size_t levenshtein::generic_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    WeightTable weights)
{
    utils::remove_common_affix(sentence1, sentence2);
    if (sentence1.size() > sentence2.size()) {
        std::swap(sentence1, sentence2);
        std::swap(weights.insert_cost, weights.delete_cost);
    }

    std::vector<std::size_t> cache(sentence1.size() + 1);

    cache[0] = 0;
    for (std::size_t i = 1; i < cache.size(); ++i) {
        cache[i] = cache[i - 1] + weights.delete_cost;
    }

    for (const auto& char2 : sentence2) {
        auto cache_iter = cache.begin();
        std::size_t temp = *cache_iter;
        *cache_iter += weights.insert_cost;

        for (const auto& char1 : sentence1) {
            if (char1 != char2) {
                temp = std::min({ *cache_iter + weights.delete_cost,
                    *(cache_iter + 1) + weights.insert_cost,
                    temp + weights.replace_cost });
            }
            ++cache_iter;
            std::swap(*cache_iter, temp);
        }
    }

    return cache.back();
}

template <typename CharT>
std::size_t levenshtein::generic_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    WeightTable weights)
{
    return generic_distance(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2),
        weights);
}

template<typename CharT>
double levenshtein::normalized_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    double min_ratio)
{
    if (sentence1.empty() || sentence2.empty()) {
        return sentence1.empty() && sentence2.empty();
    }

    std::size_t sentence1_len = sentence1.length();
    std::size_t sentence2_len = sentence2.length();
    std::size_t max_len = std::max(sentence1_len, sentence2_len);

    // constant time calculation to find a string ratio based on the string length
    // so it can exit early without running any levenshtein calculations
    std::size_t min_distance = (sentence1_len > sentence2_len)
        ? sentence1_len - sentence2_len
        : sentence2_len - sentence1_len;

    double len_ratio = 1.0 - static_cast<double>(min_distance) / max_len;
    if (len_ratio < min_ratio) {
        return 0.0;
    }

    std::size_t dist = distance(sentence1, sentence2);

    double ratio = 1.0 - static_cast<double>(dist) / max_len;
    return (ratio >= min_ratio) ? ratio : 0.0;
}

template <typename CharT>
double levenshtein::normalized_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    double min_ratio)
{
    return normalized_distance(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2),
        min_ratio);
}


template<typename CharT>
double levenshtein::normalized_weighted_distance(
    boost::basic_string_view<CharT> sentence1,
    boost::basic_string_view<CharT> sentence2,
    double min_ratio)
{
    if (sentence1.empty() || sentence2.empty()) {
        return sentence1.empty() && sentence2.empty();
    }

    std::size_t sentence1_len = sentence1.length();
    std::size_t sentence2_len = sentence2.length();
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

    std::size_t dist =  weighted_distance(sentence1, sentence2);

    if (dist > lensum) {
        return 0.0;
    }
    double ratio = 1.0 - static_cast<double>(dist) / lensum;
    return (ratio >= min_ratio) ? ratio : 0.0;
}

template <typename CharT>
double levenshtein::normalized_weighted_distance(
    const std::basic_string<CharT>& sentence1,
    const std::basic_string<CharT>& sentence2,
    double min_ratio)
{
    return normalized_weighted_distance(
        boost::basic_string_view<CharT>(sentence1),
        boost::basic_string_view<CharT>(sentence2),
        min_ratio);
}
