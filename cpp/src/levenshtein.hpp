#pragma once
#include <string_view>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <optional>
#include <numeric>
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


    template<typename MinDistanceCalc=std::false_type, typename CharT>
    auto levenshtein_word_cmp(const CharT &letter_cmp, const string_view_vec<CharT> &words,
                            std::vector<std::size_t> &cache, std::size_t current_cache);

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
    template<typename CharT, typename MinDistance=std::nullopt_t>
    std::size_t weighted_distance_impl(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2, MinDistance max_distance=std::nullopt);

    template<typename MinDistance=std::nullopt_t>
    std::size_t weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, MinDistance max_distance=std::nullopt);

    template<typename MinDistance=std::nullopt_t>
    std::size_t weighted_distance(std::string_view sentence1, std::string_view sentence2, MinDistance max_distance=std::nullopt);

    template<typename CharT, typename MinDistance=std::nullopt_t>
    std::size_t weighted_distance(string_view_vec<CharT> sentence1, string_view_vec<CharT> sentence2, MinDistance max_distance=std::nullopt);

    /**
    * Calculates a normalized score of the weighted Levenshtein algorithm between 0.0 and
    * 1.0 (inclusive), where 1.0 means the sequences are the same.
    */
    template<typename Sentence1, typename Sentence2>
    float normalized_weighted_distance(const Sentence1 &sentence1, const Sentence2 &sentence2, float min_ratio=0.0);
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
  auto m = matrix(sentence1, sentence2);
  std::size_t matrix_columns = m.matrix_columns;
  std::size_t matrix_rows = m.matrix_rows;
  std::size_t prefix_len = m.prefix_len;
  auto lev_matrix = m.matrix;

  std::vector<EditOp> ops;
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
    EditType op_type;

    if (i && j && is_replace(position)) {
      op_type = EditType::EditReplace;
      --i;
      --j;
      position -= matrix_rows + 1;
    } else if (j && is_insert(position)) {
      op_type = EditType::EditInsert;
      --j;
      --position;
    }  else if (i && is_delete(position)) {
      op_type = EditType::EditDelete;
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


template<typename MinDistanceCalc, typename CharT>
inline auto levenshtein::levenshtein_word_cmp(const CharT &letter_cmp, const string_view_vec<CharT> &words,
                          std::vector<std::size_t> &cache, std::size_t current_cache)
{
  std::size_t result = current_cache + 1;
  auto cache_iter = cache.begin();
  auto word_iter = words.begin();
  auto min_distance = std::numeric_limits<std::size_t>::max();

  auto charCmp = [&] (const CharT &char2) {
	  if (letter_cmp == char2) { result = current_cache; }
	  else { ++result; }

    current_cache = *cache_iter;
    if (result > current_cache + 1) {
      result = current_cache + 1;
    }

    if constexpr(!std::is_same_v<std::false_type, MinDistanceCalc>) {
      if (current_cache < min_distance) {
        min_distance = current_cache;
      }
    }

    *cache_iter = result;
    ++cache_iter;
  };

  // no whitespace should be added in front of the first word
  for (const auto &letter : *word_iter) {
	  charCmp(letter);
  }
  ++word_iter;

  for (; word_iter != words.end(); ++word_iter) {
    // between every word there should be one whitespace
    charCmp(0x20);

    // check following word
    for (const auto &letter : *word_iter) {
	    charCmp(letter);
    }
  }

  if constexpr(!std::is_same_v<std::false_type, MinDistanceCalc>) {
    return min_distance;
  }
}


template<typename CharT, typename MinDistance>
inline std::size_t levenshtein::weighted_distance(string_view_vec<CharT> sentence1, string_view_vec<CharT> sentence2, MinDistance max_distance) {
  remove_common_affix(sentence1, sentence2);
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
  for (const auto &letter : *word_iter) {
    if constexpr(!std::is_same_v<MinDistance, std::nullopt_t>) {
      size_t min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos);
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
    if constexpr(!std::is_same_v<MinDistance, std::nullopt_t>) {
      size_t min_distance = levenshtein_word_cmp<std::true_type>((CharT)0x20, sentence2, cache, range1_pos);
      if (min_distance > max_distance) {
        return std::numeric_limits<std::size_t>::max();
      }
    } else {
      levenshtein_word_cmp((CharT)0x20, sentence2, cache, range1_pos);
    }

    ++range1_pos;

    for (const auto &letter : *word_iter) {
      if constexpr(!std::is_same_v<MinDistance, std::nullopt_t>) {
        size_t min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos);
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


template<typename MinDistance>
inline std::size_t levenshtein::weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, MinDistance max_distance) {
  return weighted_distance_impl(sentence1, sentence2, max_distance);
}


template<typename MinDistance>
inline std::size_t levenshtein::weighted_distance(std::string_view sentence1, std::string_view sentence2, MinDistance max_distance) {
  return weighted_distance_impl(sentence1, sentence2, max_distance);
}


template<typename CharT, typename MinDistance>
inline std::size_t levenshtein::weighted_distance_impl(std::basic_string_view<CharT> sentence1, std::basic_string_view<CharT> sentence2, MinDistance max_distance) {

  remove_common_affix(sentence1, sentence2);

  if (sentence2.size() > sentence1.size()) std::swap(sentence1, sentence2);

  if (sentence2.empty()) {
    return sentence1.length();
  }

  std::vector<std::size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  std::size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto cache_iter = cache.begin();
    std::size_t current_cache = sentence1_pos;
    std::size_t result = sentence1_pos+1;
    auto min_distance = std::numeric_limits<std::size_t>::max();
    for (const auto &char2 : sentence2) {
      if (char1 == char2) {
        result = current_cache;
      } else {
        ++result;
      }
      current_cache = *cache_iter;
      if (result > current_cache + 1) {
        result = current_cache + 1;
      }
      if constexpr(!std::is_same_v<MinDistance, std::nullopt_t>) {
        if (current_cache < min_distance) {
          min_distance = current_cache;
        }
      }
      *cache_iter = result;
      ++cache_iter;
    }
    if constexpr(!std::is_same_v<MinDistance, std::nullopt_t>) {
      if (min_distance > max_distance) {
        return std::numeric_limits<std::size_t>::max();
      }
    }
    ++sentence1_pos;
  }
  return cache.back();
}



template<typename Sentence1, typename Sentence2>
inline float levenshtein::normalized_weighted_distance(const Sentence1 &sentence1, const Sentence2 &sentence2, float min_ratio)
{
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  if (sentence1.empty() || sentence1.empty()) {
    return 0.0;
  }

  std::size_t sentence1_len = utils::joined_size(sentence1);
  std::size_t sentence2_len = utils::joined_size(sentence2);
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
    ? weighted_distance(sentence1, sentence2, std::ceil((float)lensum - min_ratio * lensum))
    : weighted_distance(sentence1, sentence2);

  if (distance == std::numeric_limits<std::size_t>::max()) {
      return 0.0;
  }
  return 1.0 - (float)distance / (float)lensum;
}
