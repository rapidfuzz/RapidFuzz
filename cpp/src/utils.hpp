#pragma once
#include <string_view>
#include <vector>
#include <algorithm>
#include <iostream>

inline std::vector<std::wstring_view> splitSV(std::wstring_view str) {
  std::vector<std::wstring_view> output;
  // assume a word length of 6 + 1 whitespace
  output.reserve(str.size() / 7);

  for (auto first = str.data(), second = str.data(), last = first + str.size();
      second != last && first != last; first = second + 1) {

    second = std::find_if(first, last, [](unsigned char c){ return std::iswspace(c); });

    if (first != second)
      output.emplace_back(first, second - first);
  }

  return output;
}

struct Decomposition {
  std::vector<std::wstring_view> intersection;
  std::vector<std::wstring_view> difference_ab;
  std::vector<std::wstring_view> difference_ba;
};


inline Decomposition set_decomposition(std::vector<std::wstring_view> a, std::vector<std::wstring_view> b) {
  std::vector<std::wstring_view> intersection;
  std::vector<std::wstring_view> difference_ab;
  a.erase(std::unique(a.begin(), a.end()), a.end());
  b.erase(std::unique(b.begin(), b.end()), b.end());
  
  for (const auto &current_a : a) {
    auto element_b = std::find(b.begin(), b.end(), current_a);
    if (element_b != b.end()) {
      b.erase(element_b);
      intersection.emplace_back(current_a);
    } else {
      difference_ab.emplace_back(current_a);
    }
  }

  return Decomposition{intersection, difference_ab, b};
}


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
inline std::size_t remove_common_prefix(std::wstring_view& a, std::wstring_view& b) {
  auto prefix = common_prefix_length(a.begin(), a.end(), b.begin(), b.end());
	a.remove_prefix(prefix);
	b.remove_prefix(prefix);
  return prefix;
}

/**
 * Removes common suffix of two string views
 */
inline std::size_t remove_common_suffix(std::wstring_view& a, std::wstring_view& b) {
  auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
	a.remove_suffix(suffix);
  b.remove_suffix(suffix);
  return suffix;
}

struct Affix {
  std::size_t prefix_len;
  std::size_t suffix_len;
};

/**
 * Removes common affix of two string views
 */
inline Affix remove_common_affix(std::wstring_view& a, std::wstring_view& b) {
	return Affix {
    remove_common_prefix(a, b),
    remove_common_suffix(a, b)
  };
}


template<typename T>
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

template<typename T>
inline void remove_common_affix(std::vector<T> &a, std::vector<T> &b)
{
  vec_remove_common_affix(a, b);
  if (!a.empty() && !b.empty()) {
    remove_common_prefix(a.front(), b.front());
    remove_common_suffix(a.back(), b.back());
  }
}


template<typename T>
inline std::size_t recursiveIterableSize(const T &x, std::size_t delimiter_length=0){
	return x.size();
}

template<typename T>
inline std::size_t recursiveIterableSize(const std::vector<T> &x, std::size_t delimiter_length=0){
  if (x.empty()) {
    return 0;
  }
	std::size_t result = (x.size() - 1) * delimiter_length;
	for (const auto &y: x) {
		result += recursiveIterableSize(y, delimiter_length);
	}
	return result;
}


inline std::wstring sentence_join(const std::vector<std::wstring_view> &sentence) {
  if (sentence.empty()) {
    return L"";
  }

  auto sentence_iter = sentence.begin();
  std::wstring result = std::wstring {*sentence_iter};
  ++sentence_iter;
  for (; sentence_iter != sentence.end(); ++sentence_iter) {
    result.append(L" ").append(std::wstring {*sentence_iter});
  }
  return result;
}