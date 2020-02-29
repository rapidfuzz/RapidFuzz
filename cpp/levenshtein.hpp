#pragma once
#include "utils.hpp"
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>


void levenshtein_word_cmp(const char &letter_cmp,
                          const std::vector<std::string_view> &words,
                          std::vector<size_t> &cache, size_t distance_b)
{
  size_t result = distance_b + 1;
  auto cache_iter = cache.begin();
  auto word_iter = words.begin();

  auto charCmp = [&] (const char &char2) {
	if (letter_cmp == char2) { result = distance_b - 1; }
	else { ++result; }

    distance_b = *cache_iter;
    if (result > distance_b + 1) {
      result = distance_b + 1;
    }
    *cache_iter = result;
    ++cache_iter;
  };

  // words | view::join(' ') would be a bit nicer to write here but is a lot slower
  // might be worth a retry when it is added in c++20 since then compilers might
  // improve the runtime

  // no whitespace should be added in front of the first word
  for (const auto &letter : *word_iter) {
	charCmp(letter);
  }
  ++word_iter;

  for (; word_iter != words.end(); ++word_iter) {
    // between every word there should be a whitespace
	charCmp(' ');
    // check following word
    for (const auto &letter : *word_iter) {
	  charCmp(letter);
    }
  }
}


size_t levenshtein(std::vector<std::string_view> sentence1,
                   std::vector<std::string_view> sentence2) {
  remove_common_affix(sentence1, sentence2);
  size_t sentence1_len = joinedStringViewLength(sentence1);
  size_t sentence2_len = joinedStringViewLength(sentence2);

  // exit early when one sentence is empty
  // (empty sentence would cause undefinded behaviour)
  if (!sentence1_len) {
    return sentence2_len;
  }
  if (!sentence2_len) {
    return sentence1_len;
  }

  std::vector<size_t> cache(sentence2_len);
  std::iota(cache.begin(), cache.end(), 1);

  size_t range1_pos = 0;
  auto word1_iter = sentence1.begin();

  // no whitespace in front of first word
  size_t distance_b = range1_pos;
  for (const auto &letter : *word1_iter) {
    distance_b = range1_pos;
    levenshtein_word_cmp(letter, sentence2, cache, distance_b);
    ++range1_pos;
  }

  ++word1_iter;
  for (; word1_iter != sentence1.end(); ++word1_iter) {
    distance_b = range1_pos;

    // whitespace between words
    distance_b = range1_pos;
    levenshtein_word_cmp(' ', sentence2, cache, distance_b);
    ++range1_pos;

    for (const auto &letter : *word1_iter) {
      distance_b = range1_pos;
      levenshtein_word_cmp(letter, sentence2, cache, distance_b);
      ++range1_pos;
    }
  }
  return cache.back();
}


float normalized_levenshtein(std::vector<std::string_view> sentence1,
                             std::vector<std::string_view> sentence2) {
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = joinedStringViewLength(sentence1) + joinedStringViewLength(sentence2);
  size_t distance = levenshtein(sentence1, sentence2);
  return 1.0 - (float)distance / (float)lensum;
}


size_t levenshtein(std::string_view sentence1, std::string_view sentence2) {
  remove_common_affix(sentence1, sentence2);

  if (sentence1.empty()) {
    return sentence2.length();
  }
  if (sentence2.empty()) {
    return sentence1.length();
  }

  std::vector<size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  size_t sentence1_pos = 0;
  size_t result = sentence2.length();
  for (const auto &char1 : sentence1) {
    size_t distance_b = sentence1_pos;
    result = sentence1_pos + 1;
    auto cache_iter = cache.begin();
    for (const auto &char2 : sentence2) {
      if (char1 == char2) {
        result = distance_b - 1;
      } else {
        ++result;
      }
      distance_b = *cache_iter;
      if (result > distance_b + 1) {
        result = distance_b + 1;
      }
      *cache_iter = result;
      ++cache_iter;
    }
    ++sentence1_pos;
  }
  return result;
}


float normalized_levenshtein(std::string_view sentence1, std::string_view sentence2) {
  if (sentence1.empty() && sentence2.empty()) {
    return 1.0;
  }

  size_t lensum = sentence1.length() + sentence2.length();
  size_t distance = levenshtein(sentence1, sentence2);
  return 1.0 - (float)distance / (float)lensum;
}