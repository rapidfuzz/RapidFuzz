#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <locale>


template<typename CharT>
using string_view_vec = std::vector<std::basic_string_view<CharT>>;


namespace detail {
    template<typename T>
    auto char_type(T const*) -> T;

    template<typename T, typename U = typename T::const_iterator>
    auto char_type(T const&) -> typename std::iterator_traits<U>::value_type;
}

template<typename T>
using char_type = decltype(detail::char_type(std::declval<T const&>()));


template<typename CharT>
struct DecomposedSet {
  string_view_vec<CharT> intersection;
  string_view_vec<CharT> difference_ab;
  string_view_vec<CharT> difference_ba;
  DecomposedSet(string_view_vec<CharT> intersection, string_view_vec<CharT> difference_ab, string_view_vec<CharT> difference_ba)
    : intersection(std::move(intersection)), difference_ab(std::move(difference_ab)), difference_ba(std::move(difference_ba)) {}
};


namespace utils {

  template<
      typename T, typename CharT = char_type<T>,
      typename = std::enable_if_t<std::is_convertible<T const&, std::basic_string_view<CharT>>{}>
  >
  string_view_vec<CharT> splitSV(const T &str);


  template<typename CharT>
  DecomposedSet<CharT> set_decomposition(string_view_vec<CharT> a, string_view_vec<CharT> b);


  template<typename T>
  std::size_t joined_size(const T &x);

  template<typename T>
  std::size_t joined_size(const std::vector<T> &x);


  template<typename CharT>
  std::basic_string<CharT> join(const string_view_vec<CharT> &sentence);
}


template<typename T, typename CharT, typename>
string_view_vec<CharT> utils::splitSV(const T &str) {
  string_view_vec<CharT> output;
  // assume a word length of 6 + 1 whitespace
  output.reserve(str.size() / 7);

  auto first = str.data(), second = str.data(), last = first + str.size();
  for (; second != last && first != last; first = second + 1) {
    // maybe use localisation
    second = std::find_if(first, last, [](unsigned char c){ return std::isspace(c); });

    if (first != second)
      output.emplace_back(first, second - first);
  }

  return output;
}


template<typename CharT>
DecomposedSet<CharT> utils::set_decomposition(string_view_vec<CharT> a, string_view_vec<CharT> b) {
  string_view_vec<CharT> intersection;
  string_view_vec<CharT> difference_ab;
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

  return DecomposedSet{intersection, difference_ab, b};
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
template<typename CharT>
inline std::size_t remove_common_prefix(std::basic_string_view<CharT>& a, std::basic_string_view<CharT>& b) {
  auto prefix = common_prefix_length(a.begin(), a.end(), b.begin(), b.end());
	a.remove_prefix(prefix);
	b.remove_prefix(prefix);
  return prefix;
}

/**
 * Removes common suffix of two string views
 */
template<typename CharT>
inline std::size_t remove_common_suffix(std::basic_string_view<CharT>& a, std::basic_string_view<CharT>& b) {
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
template<typename CharT>
inline Affix remove_common_affix(std::basic_string_view<CharT>& a, std::basic_string_view<CharT>& b) {
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
inline std::size_t utils::joined_size(const T &x){
	return x.size();
}


template<typename T>
inline std::size_t utils::joined_size(const std::vector<T> &x){
  if (x.empty()) {
    return 0;
  }

  // there is a whitespace between each word
  std::size_t result = x.size() - 1;
	for (const auto &y: x) result += y.size();

	return result;
}


template<typename CharT>
std::basic_string<CharT> utils::join(const string_view_vec<CharT> &sentence) {
  if (sentence.empty()) {
    return std::basic_string<CharT>();
  }

  auto sentence_iter = sentence.begin();
  std::basic_string<CharT> result {*sentence_iter};
  const std::basic_string<CharT> whitespace {0x20};
  ++sentence_iter;
  for (; sentence_iter != sentence.end(); ++sentence_iter) {
    result.append(whitespace).append(std::basic_string<CharT> {*sentence_iter});
  }
  return result;
}