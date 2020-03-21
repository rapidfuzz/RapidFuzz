#pragma once
#include <string_view>
#include <vector>
#include <cmath>
#include <stdexcept>
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

    template<typename CharT>
    Matrix matrix(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2);

    template<typename CharT>
    std::vector<EditOp> editops(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2);

    struct MatchingBlock {
    	std::size_t first_start;
    	std::size_t second_start;
    	std::size_t len;
      MatchingBlock(std::size_t first_start, std::size_t second_start, std::size_t len)
        : first_start(first_start), second_start(second_start), len(len) {}
    };

    template<typename CharT>
    std::vector<MatchingBlock> matching_blocks(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2);


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
    std::size_t weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2,
                             std::wstring_view delimiter=L"");
    std::size_t weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2,
                             std::wstring_view delimiter=L"");


    /**
     * These functions allow providing a max_distance parameter that can be used to exit early when the
     * calculated levenshtein distance is at least as big as max_distance and will return the maximal
     * possible value for std::size_t.
     * This range check makes the levenshtein calculation about 20% slower, so it should be only used
     * when it can usually exit early.
     */
    std::size_t weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2,
                             std::size_t max_distance, std::wstring_view delimiter=L"");
    std::size_t weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2,
                             std::size_t max_distance, std::wstring_view delimiter=L"");

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    template<typename Sentence1, typename Sentence2>
    float normalized_weighted_distance(const Sentence1 &sentence1, const Sentence2 &sentence2,
                                       float min_ratio=0.0, std::wstring_view delimiter=L"")
    {
        if (sentence1.empty() && sentence2.empty()) {
            return 1.0;
        }

        if (sentence1.empty() || sentence1.empty()) {
          return 0.0;
        }

        std::size_t sentence1_len = utils::joined_size(sentence1, delimiter);
        std::size_t sentence2_len = utils::joined_size(sentence2, delimiter);
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



template<typename CharT>
inline levenshtein::Matrix levenshtein::matrix(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2) {
  Affix affix = remove_common_affix(sentence1, sentence2);

  std::size_t matrix_columns = sentence1.length() + 1;
  std::size_t matrix_rows = sentence2.length() + 1;

  std::vector<std::size_t> cache_matrix(matrix_rows*matrix_columns, 0);

  for (std::size_t i = 0; i < matrix_rows; ++i) {
    cache_matrix[i] = i;
  }

  for (std::size_t i = 1; i < matrix_columns; ++i) {
    cache_matrix[matrix_rows*i] = i;
  }

  std::size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto prev_cache = cache_matrix.begin() + sentence1_pos * matrix_rows;
    auto result_cache = cache_matrix.begin() + (sentence1_pos + 1) * matrix_rows + 1;
    std::size_t result = sentence1_pos + 1;
    for (const auto &char2 : sentence2) {
      result = std::min({
        result + 1,
        *prev_cache + (char1 != char2),
        *(++prev_cache) + 1
      });
      *result_cache = result;
      ++result_cache;
    }
    ++sentence1_pos;
  }

  return Matrix {
      affix.prefix_len,
      cache_matrix,
      matrix_columns,
      matrix_rows
  };
}


template<typename CharT>
inline std::vector<levenshtein::EditOp>
levenshtein::editops(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2) {
  auto lev_matrix = matrix(sentence1, sentence2);
  std::size_t matrix_columns = lev_matrix.matrix_columns;
  std::size_t matrix_rows = lev_matrix.matrix_rows;
  std::size_t prefix_len = lev_matrix.prefix_len;
  auto matrix = lev_matrix.matrix;

  std::vector<EditOp> ops;
  ops.reserve(matrix[matrix_columns * matrix_rows - 1]);

  std::size_t i = matrix_columns - 1;
  std::size_t j = matrix_rows - 1;
  std::size_t pos = matrix_columns * matrix_rows - 1;

  auto is_replace = [=](std::size_t pos) {
    return matrix[pos - matrix_rows - 1] < matrix[pos];
  };
  auto is_insert = [=](std::size_t pos) {
    return matrix[pos - 1] < matrix[pos];
  };
  auto is_delete = [=](std::size_t pos) {
    return matrix[pos - matrix_rows] < matrix[pos];
  };
  auto is_keep = [=](std::size_t pos) {
    return matrix[pos - matrix_rows - 1] == matrix[pos];
  };

  while (i > 0 || j > 0) {
    EditType op_type;

    if (i && j && is_replace(pos)) {
      op_type = EditType::EditReplace;
      --i;
      --j;
      pos -= matrix_rows + 1;
    } else if (j && is_insert(pos)) {
      op_type = EditType::EditInsert;
      --j;
      --pos;
    }  else if (i && is_delete(pos)) {
      op_type = EditType::EditDelete;
      --i;
      pos -= matrix_rows;
    } else if (is_keep(pos)) {
      --i;
      --j;
      pos -= matrix_rows + 1;
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
inline std::vector<levenshtein::MatchingBlock>
levenshtein::matching_blocks(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2) {
  auto edit_ops = editops(sentence1, sentence2);
  std::size_t first_start = 0;
	std::size_t second_start = 0;
  std::vector<MatchingBlock> mblocks;

  for (const auto &op : edit_ops) {
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