#include "levenshtein.hpp"
#include <numeric>
#include <optional>


template<typename MinDistanceCalc=std::false_type, typename CharT, typename Delimiter=std::nullopt_t>
auto levenshtein_word_cmp(const char &letter_cmp, const string_view_vec<CharT> &words,
                          std::vector<std::size_t> &cache, std::size_t current_cache, Delimiter delimiter=std::nullopt)
{
  std::size_t result = current_cache + 1;
  auto cache_iter = cache.begin();
  auto word_iter = words.begin();
  auto min_distance = std::numeric_limits<std::size_t>::max();

  auto charCmp = [&] (const char &char2) {
	  if (letter_cmp == char2) { result = current_cache; }
	  else { ++result; }

    current_cache = *cache_iter;
    if (result > current_cache + 1) {
      result = current_cache + 1;
    }

    if constexpr(!std::is_same<std::false_type, MinDistanceCalc>::value) {
      if (current_cache < min_distance) {
        min_distance = current_cache;
      }
    }

    *cache_iter = result;
    ++cache_iter;
  };

  // no delimiter should be added in front of the first word
  for (const auto &letter : *word_iter) {
	  charCmp(letter);
  }
  ++word_iter;

  for (; word_iter != words.end(); ++word_iter) {
    // between every word there should be a delimiter if one exists
    if constexpr(!std::is_same<std::nullopt_t, Delimiter>::value) {
      for (const auto &letter : delimiter) {
        charCmp(letter);
      }
    }
    // check following word
    for (const auto &letter : *word_iter) {
	    charCmp(letter);
    }
  }

  if constexpr(!std::is_same<std::false_type, MinDistanceCalc>::value) {
    return min_distance;
  }
}


std::size_t levenshtein::weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2, std::wstring_view delimiter) {
  remove_common_affix(sentence1, sentence2);
  std::size_t sentence1_len = utils::joined_size(sentence1, delimiter);
  std::size_t sentence2_len = utils::joined_size(sentence2, delimiter);

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


std::size_t levenshtein::weighted_distance(std::vector<std::wstring_view> sentence1, std::vector<std::wstring_view> sentence2, std::size_t max_distance, std::wstring_view delimiter) {
  remove_common_affix(sentence1, sentence2);
  std::size_t sentence1_len = utils::joined_size(sentence1, delimiter);
  std::size_t sentence2_len = utils::joined_size(sentence2, delimiter);

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
    auto min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos, delimiter);
    if (min_distance > max_distance) {
      return std::numeric_limits<std::size_t>::max();
    }
    ++range1_pos;
  }

  ++word_iter;
  for (; word_iter != sentence1.end(); ++word_iter) {
    // delimiter between words
    for (const auto &letter : delimiter) {
      auto min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos, delimiter);
      if (min_distance > max_distance) {
        return std::numeric_limits<std::size_t>::max();
      }
      ++range1_pos;
    }

    for (const auto &letter : *word_iter) {
      auto min_distance = levenshtein_word_cmp<std::true_type>(letter, sentence2, cache, range1_pos, delimiter);
      if (min_distance > max_distance) {
        return std::numeric_limits<std::size_t>::max();
      }
      ++range1_pos;
    }
  }

  return cache.back();
}


std::size_t levenshtein::weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, std::wstring_view delimiter) {
  remove_common_affix(sentence1, sentence2);

  if (sentence2.length() > sentence1.length()) std::swap(sentence1, sentence2);

  if (sentence2.empty()) {
    return sentence1.length();
  }

  std::vector<std::size_t> cache(sentence2.length());
  std::iota(cache.begin(), cache.end(), 1);

  std::size_t sentence1_pos = 0;
  for (const auto &char1 : sentence1) {
    auto cache_iter = cache.begin();
    std::size_t current_cache = sentence1_pos;
    std::size_t result = sentence1_pos + 1;
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


std::size_t levenshtein::weighted_distance(std::wstring_view sentence1, std::wstring_view sentence2, std::size_t max_distance, std::wstring_view delimiter) {
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

      if (current_cache < min_distance) {
        min_distance = current_cache;
      }
      *cache_iter = result;
      ++cache_iter;
    }
    if (min_distance > max_distance) {
      return std::numeric_limits<std::size_t>::max();
    }
    ++sentence1_pos;
  }
  return cache.back();
}
