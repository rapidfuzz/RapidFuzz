#pragma once
#include <string_view>
#include <vector>
#include <algorithm>
#include "concepts.hpp"
#include <iostream>

struct Affix {
  size_t prefix_len;
  size_t suffix_len;
};

class MatchingPair {
public:
  std::string a;
  std::string b;

  MatchingPair(std::string a, std::string b) : a(a), b(b) {}

  Affix remove_affix() {
    size_t a_len = a.length();
    size_t b_len = b.length();

    while (a_len > 0 && b_len > 0 && a[a_len - 1] == b[b_len - 1]) {
      --a_len;
      --b_len;
    }

    size_t prefix_len = 0;
    while (a_len > 0 && b_len > 0 && a[prefix_len] == b[prefix_len]) {
      --a_len;
      --b_len;
      ++prefix_len;
    }

    size_t suffix_len = a.length() - a_len;
    a = a.substr(prefix_len, a_len);
    b = b.substr(prefix_len, b_len);
    return Affix{prefix_len, suffix_len};
  }
};

// construct a sorted range both for intersections and differences between
// sorted ranges based on reference implementations of set_difference and
// set_intersection http://www.cplusplus.com/reference/algorithm/set_difference/
// http://www.cplusplus.com/reference/algorithm/set_intersection/
template <class InputIterator1, class InputIterator2, class OutputIterator1,
          class OutputIterator2, class OutputIterator3>
inline OutputIterator3 set_decomposition(InputIterator1 first1, InputIterator1 last1,
                                  InputIterator2 first2, InputIterator2 last2,
                                  OutputIterator1 result1,
                                  OutputIterator2 result2,
                                  OutputIterator3 result3) {
  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2) {
      *result1++ = *first1++;
    } else if (*first2 < *first1) {
      *result2++ = *first2++;
    } else {
      *result3++ = *first1++;
      ++first2;
    }
  }
  std::copy(first1, last1, result1);
  std::copy(first2, last2, result2);
  return result3;
}

inline std::vector<std::string_view> splitSV(std::string_view str,
                                      std::string_view delims = " ") {
  std::vector<std::string_view> output;
  // assume a word length of 6 + 1 whitespace
  output.reserve(str.size() / 7);

  for (auto first = str.data(), second = str.data(), last = first + str.size();
       second != last && first != last; first = second + 1) {
    second =
        std::find_first_of(first, last, std::cbegin(delims), std::cend(delims));

    if (first != second)
      output.emplace_back(first, second - first);
  }

  return output;
}

struct Intersection {
  std::vector<std::string_view> sect;
  std::vector<std::string_view> ab;
  std::vector<std::string_view> ba;
};


inline Intersection intersection_count_sorted_vec(std::vector<std::string_view> a,
                                           std::vector<std::string_view> b) {
  std::vector<std::string_view> vec_sect;
  std::vector<std::string_view> vec_ab;
  a.erase(std::unique(a.begin(), a.end()), a.end());
  b.erase(std::unique(b.begin(), b.end()), b.end());
  
  for (const auto &current_a : a) {
    auto element_b = std::find(b.begin(), b.end(), current_a);
    if (element_b != b.end()) {
      b.erase(element_b);
      vec_sect.emplace_back(current_a);
    } else {
      vec_ab.emplace_back(current_a);
    }
  }

  return Intersection{vec_sect, vec_ab, b};
}

/*
this needs more work to make it generic
*/


/**
 * Finds the longest common prefix between two ranges
 */
template <typename InputIterator1, typename InputIterator2>
inline auto common_prefix_length(InputIterator1 first1, InputIterator1 last1,
	                        InputIterator2 first2, InputIterator2 last2)
{
    return std::distance(first1, std::mismatch(first1, last1, first2, last2).first);
}

/**
 * Removes common prefix of two string views
 */
inline size_t remove_common_prefix(std::string_view& a, std::string_view& b) {
  auto prefix = common_prefix_length(a.begin(), a.end(), b.begin(), b.end());
	a.remove_prefix(prefix);
	b.remove_prefix(prefix);
  return prefix;
}

/**
 * Removes common suffix of two string views
 */
inline size_t remove_common_suffix(std::string_view& a, std::string_view& b) {
  auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
	a.remove_suffix(suffix);
  b.remove_suffix(suffix);
  return suffix;
}

/**
 * Removes common affix of two string views
 */
inline Affix remove_common_affix(std::string_view& a, std::string_view& b) {
	return Affix {
    remove_common_prefix(a, b),
    remove_common_suffix(a, b)
  };
}


template<Iterable T>
inline void vec_remove_common_affix(T &a, T &b) {
  auto prefix = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
  a.erase(a.begin(), prefix.first);
  b.erase(b.begin(), prefix.second);
  auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
  a.erase(a.end()-suffix, a.end());
  b.erase(b.end()-suffix, b.end());
}

template<typename T>
inline void vec_common_affix(std::vector<T> &a, std::vector<T> &b) {
  iterable_remove_common_affix(a, b);
}

template<Iterable T>
inline void remove_common_affix(std::vector<T> &a, std::vector<T> &b)
{
  vec_remove_common_affix(a, b);
  if (!a.empty() && !b.empty()) {
    remove_common_prefix(a.front(), b.front());
    remove_common_suffix(a.back(), b.back());
  }
}


template<Iterable T>
inline size_t recursiveIterableSize(const T &x, size_t delimiter_length=0){
	return x.size();
}

template<Iterable T>
inline size_t recursiveIterableSize(const std::vector<T> &x, size_t delimiter_length=0){
  if (x.empty()) {
    return 0;
  }
	size_t result = (x.size() - 1) * delimiter_length;
	for (const auto &y: x) {
		result += recursiveIterableSize(y, delimiter_length);
	}
	return result;
}


inline std::string sentenceJoin(const std::vector<std::string_view> &sentence) {
  if (sentence.empty()) {
    return "";
  }

  auto sentence_iter = sentence.begin();
  std::string result = std::string {*sentence_iter};
  ++sentence_iter;
  for (; sentence_iter != sentence.end(); ++sentence_iter) {
    result +=  " " + std::string {*sentence_iter};
  }
  return result;
}