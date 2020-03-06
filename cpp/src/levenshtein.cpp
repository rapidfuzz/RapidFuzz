#include "levenshtein.hpp"
#include "utils.hpp"
#include <numeric>
#include <cmath>
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

  // exit early when one sentence is empty
  // would cause problems in algorithm
  if (!sentence1_len) {
    return sentence2_len;
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

  // exit early when one sentence is empty
  // would cause problems in algorithm
  if (!sentence1_len) {
    return sentence2_len;
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


size_t levenshtein::weighted_distance(std::string_view sentence1, std::string_view sentence2) {
  remove_common_affix(sentence1, sentence2);
  if (sentence1.empty()) {
    return sentence2.length();
  }
  if (sentence2.empty()) {
    return sentence1.length();
  }

  if (sentence2.length() > sentence1.length()) std::swap(sentence1, sentence2);

  std::vector<size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto cache_iter = cache.begin();
    size_t current_cache = sentence1_pos;
    size_t result = sentence1_pos+1;
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


size_t levenshtein::weighted_distance(std::string_view sentence1, std::string_view sentence2, size_t max_distance) {
  remove_common_affix(sentence1, sentence2);
  if (sentence1.empty()) {
    return sentence2.length();
  }
  if (sentence2.empty()) {
    return sentence1.length();
  }

  if (sentence2.length() > sentence1.length()) std::swap(sentence1, sentence2);

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



float levenshtein::normalized_weighted_distance(std::vector<std::string_view> sentence1,
                                                std::vector<std::string_view> sentence2,
                                                std::string_view delimiter) {
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = recursiveIterableSize(sentence1, delimiter.length()) + recursiveIterableSize(sentence2, delimiter.length());
  size_t distance = weighted_distance(sentence1, sentence2, delimiter);
  return 1.0 - (float)distance / (float)lensum;
}


float levenshtein::normalized_weighted_distance(std::string_view sentence1, std::string_view sentence2) {
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = sentence1.length() + sentence2.length();
  size_t distance = weighted_distance(sentence1, sentence2);
  return 1.0 - (float)distance / (float)lensum;
}


float levenshtein::normalized_weighted_distance(std::vector<std::string_view> sentence1, std::vector<std::string_view> sentence2,
                                   float min_ratio, std::string_view delimiter)
{
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = recursiveIterableSize(sentence1, delimiter.length()) + recursiveIterableSize(sentence2, delimiter.length());
  size_t min_distance = static_cast<size_t>(std::ceil((float)lensum - min_ratio * lensum));
  size_t distance = weighted_distance(sentence1, sentence2, min_distance, delimiter);
  if (distance == std::numeric_limits<size_t>::max()) {
    return 0.0;
  }
  return 1.0 - (float)distance / (float)lensum;
}


float levenshtein::normalized_weighted_distance(std::string_view sentence1, std::string_view sentence2, float min_ratio) {
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = sentence1.length() + sentence2.length();
  size_t min_distance = static_cast<size_t>(std::ceil((float)lensum - min_ratio * lensum));
  size_t distance = weighted_distance(sentence1, sentence2, min_distance);
  if (distance == std::numeric_limits<size_t>::max()) {
    return 0.0;
  }
  return 1.0 - (float)distance / (float)lensum;
}

