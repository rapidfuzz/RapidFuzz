#include "levenshtein.hpp"
#include <numeric>
#include <iostream>


levenshtein::Matrix levenshtein::matrix(std::string_view sentence1, std::string_view sentence2) {
  Affix affix = remove_common_affix(sentence1, sentence2);

  size_t matrix_columns = sentence1.length() + 1;
  size_t matrix_rows = sentence2.length() + 1;

  std::vector<size_t> cache_matrix(matrix_rows*matrix_columns, 0);

  for (size_t i = 0; i < matrix_rows; ++i) {
    cache_matrix[i] = i;
  }

  for (size_t i = 1; i < matrix_columns; ++i) {
    cache_matrix[matrix_rows*i] = i;
  }

  size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto prev_cache = cache_matrix.begin() + sentence1_pos * matrix_rows;
    auto result_cache = cache_matrix.begin() + (sentence1_pos + 1) * matrix_rows + 1;
    size_t result = sentence1_pos + 1;
    size_t sentence2_pos = 0;
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

/*eturn editops_from_cost_matrix(matrix_columns, matrix_rows,
                                  affix.prefix_len, cache_matrix);*/
/*
std::vector<EditOp> editops_from_cost_matrix(size_t len1,
                                             size_t len2, size_t prefix_len,
                                             std::vector<size_t> cache_matrix) {*/
                                             /*
                                             struct Matrix {
        size_t prefix_len;
        std::vector<size_t> matrix;
        size_t matrix_columns;
        size_t matrix_rows;
    };*/

std::vector<levenshtein::EditOp> levenshtein::editops(std::string_view sentence1, std::string_view sentence2) {
  auto lev_matrix = matrix(sentence1, sentence2);
  size_t matrix_columns = lev_matrix.matrix_columns;
  size_t matrix_rows = lev_matrix.matrix_rows;
  size_t prefix_len = lev_matrix.prefix_len;
  auto matrix = lev_matrix.matrix;

  std::vector<EditOp> ops;
  ops.reserve(matrix[matrix_columns * matrix_rows - 1]);

  size_t i = matrix_columns - 1;
  size_t j = matrix_rows - 1;
  size_t pos = matrix_columns * matrix_rows - 1;

  auto is_replace = [=](size_t pos) {
    return matrix[pos - matrix_rows - 1] < matrix[pos];
  };
  auto is_insert = [=](size_t pos) {
    return matrix[pos - 1] < matrix[pos];
  };
  auto is_delete = [=](size_t pos) {
    return matrix[pos - matrix_rows] < matrix[pos];
  };
  auto is_keep = [=](size_t pos) {
    return matrix[pos - matrix_rows - 1] == matrix[pos];
  };
  

  while (i > 0 || j > 0) {
    size_t current_value = matrix[pos];
    EditType op_type;

    if (is_replace(pos)) {
      op_type = EditType::EditReplace;
      --i;
      --j;
      pos -= matrix_rows + 1;
    } else if (is_insert(pos)) {
      op_type = EditType::EditInsert;
      --j;
      --pos;
    }  else if (is_delete(pos)) {
      op_type = EditType::EditDelete;
      --i;
      pos -= matrix_rows;
    } else if (is_keep(pos)) {
      --i;
      --j;
      pos -= matrix_rows + 1;
      // EditKeep does not has to be stored
      continue;
    }

    EditOp edit_op{
        op_type,
        i + prefix_len,
        j + prefix_len,
    };
    // maybe place in the back instead
    ops.insert(ops.begin(), edit_op);
  }
  return ops;
}


void levenshtein_word_cmp(const char &letter_cmp,
                          const std::vector<std::string_view> &words,
                          std::vector<size_t> &cache, size_t current_cache,
                          std::string_view delimiter="")
{
  size_t result = current_cache + 1;
  auto cache_iter = cache.begin();
  auto word_iter = words.begin();

  auto charCmp = [&] (const char &char2) {
	  if (letter_cmp == char2) { result = current_cache; }
	  else { ++result; }

    current_cache = *cache_iter;
    if (result > current_cache + 1) {
      result = current_cache + 1;
    }
    *cache_iter = result;
    ++cache_iter;
  };

  // words | view::join(' ') would be a bit nicer to write here but is a lot slower
  // might be worth a retry when it is added in c++20 since then compilers might
  // improve the runtime

  // no delimiter should be added in front of the first word
  for (const auto &letter : *word_iter) {
	  charCmp(letter);
  }
  ++word_iter;

  for (; word_iter != words.end(); ++word_iter) {
    // between every word there should be a delimiter
    for (const auto &letter : delimiter) {
      charCmp(letter);
    }
    // check following word
    for (const auto &letter : *word_iter) {
	    charCmp(letter);
    }
  }
}


size_t levenshtein::weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2, std::string_view delimiter) {
  remove_common_affix(sentence1, sentence2);
  size_t sentence1_len = recursiveIterableSize(sentence1, delimiter.length());
  size_t sentence2_len = recursiveIterableSize(sentence2, delimiter.length());

  if (sentence2_len > sentence1_len) {
    std::swap(sentence1, sentence2);
    std::swap(sentence1_len, sentence2_len);
  }

  if (!sentence2_len) {
    return sentence1_len;
  }

  std::vector<size_t> cache(sentence2_len);
  std::iota(cache.begin(), cache.end(), 1);

  size_t range1_pos = 0;
  auto word_iter = sentence1.begin();

  // no delimiter in front of first word
  for (const auto &letter : *word_iter) {
    levenshtein_word_cmp(letter, sentence2, cache, range1_pos, delimiter);
    ++range1_pos;
  }

  ++word_iter;
  for (; word_iter != sentence1.end(); ++word_iter) {
    // delimiter between words
    for (const auto &letter : delimiter) {
      levenshtein_word_cmp(letter, sentence2, cache, range1_pos, delimiter);
      ++range1_pos;
    }

    for (const auto &letter : *word_iter) {
      levenshtein_word_cmp(letter, sentence2, cache, range1_pos, delimiter);
      ++range1_pos;
    }
  }

  return cache.back();
}


size_t levenshtein_word_cmp_limited(const char &letter_cmp,
                          const std::vector<std::string_view> &words,
                          std::vector<size_t> &cache, size_t current_cache,
                          std::string_view delimiter="")
{
  size_t result = current_cache + 1;
  auto cache_iter = cache.begin();
  auto word_iter = words.begin();
  auto min_distance = std::numeric_limits<size_t>::max();

  auto charCmp = [&] (const char &char2) {
	  if (letter_cmp == char2) { result = current_cache; }
	  else { ++result; }

    current_cache = *cache_iter;
    if (result > current_cache + 1) {
      result = current_cache + 1;
    }

    if (current_cache < min_distance) {
      min_distance = current_cache;
    }
    *cache_iter = result;
    ++cache_iter;
  };

  // words | view::join(' ') would be a bit nicer to write here but is a lot slower
  // might be worth a retry when it is added in c++20 since then compilers might
  // improve the runtime

  // no delimiter should be added in front of the first word
  for (const auto &letter : *word_iter) {
	  charCmp(letter);
  }
  ++word_iter;

  for (; word_iter != words.end(); ++word_iter) {
    // between every word there should be a delimiter
    for (const auto &letter : delimiter) {
      charCmp(letter);
    }
    // check following word
    for (const auto &letter : *word_iter) {
	    charCmp(letter);
    }
  }
  return min_distance;
}


size_t levenshtein::weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2, size_t max_distance, std::string_view delimiter) {
  remove_common_affix(sentence1, sentence2);
  size_t sentence1_len = recursiveIterableSize(sentence1, delimiter.length());
  size_t sentence2_len = recursiveIterableSize(sentence2, delimiter.length());

  if (sentence2_len > sentence1_len) {
    std::swap(sentence1, sentence2);
    std::swap(sentence1_len, sentence2_len);
  }

  if (!sentence2_len) {
    return sentence1_len;
  }

  std::vector<size_t> cache(sentence2_len);
  std::iota(cache.begin(), cache.end(), 1);

  size_t range1_pos = 0;
  auto word_iter = sentence1.begin();

  // no delimiter in front of first word
  for (const auto &letter : *word_iter) {
    auto min_distance = levenshtein_word_cmp_limited(letter, sentence2, cache, range1_pos, delimiter);
    if (min_distance > max_distance) {
      return std::numeric_limits<size_t>::max();
    }
    ++range1_pos;
  }

  ++word_iter;
  for (; word_iter != sentence1.end(); ++word_iter) {
    // delimiter between words
    for (const auto &letter : delimiter) {
      auto min_distance = levenshtein_word_cmp_limited(letter, sentence2, cache, range1_pos, delimiter);
      if (min_distance > max_distance) {
        return std::numeric_limits<size_t>::max();
      }
      ++range1_pos;
    }

    for (const auto &letter : *word_iter) {
      auto min_distance = levenshtein_word_cmp_limited(letter, sentence2, cache, range1_pos, delimiter);
      if (min_distance > max_distance) {
        return std::numeric_limits<size_t>::max();
      }
      ++range1_pos;
    }
  }

  return cache.back();
}


size_t levenshtein::weighted_distance(std::string_view sentence1, std::string_view sentence2, std::string_view delimiter) {
  remove_common_affix(sentence1, sentence2);

  if (sentence2.length() > sentence1.length()) std::swap(sentence1, sentence2);

  if (sentence2.empty()) {
    return sentence1.length();
  }

  std::vector<size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto cache_iter = cache.begin();
    size_t current_cache = sentence1_pos;
    size_t result = sentence1_pos + 1;
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
      *cache_iter = result;
      ++cache_iter;
    }
    ++sentence1_pos;
  }

  return cache.back();
}


size_t levenshtein::weighted_distance(std::string_view sentence1, std::string_view sentence2, size_t max_distance, std::string_view delimiter) {
  remove_common_affix(sentence1, sentence2);

  if (sentence2.length() > sentence1.length()) std::swap(sentence1, sentence2);

  if (sentence2.empty()) {
    return sentence1.length();
  }

  std::vector<size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto cache_iter = cache.begin();
    size_t current_cache = sentence1_pos;
    size_t result = sentence1_pos+1;
    auto min_distance = std::numeric_limits<size_t>::max();
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

      if (current_cache < min_distance) {
        min_distance = current_cache;
      }
      *cache_iter = result;
      ++cache_iter;
    }
    if (min_distance > max_distance) {
      return std::numeric_limits<size_t>::max();
    }
    ++sentence1_pos;
  }
  return cache.back();
}