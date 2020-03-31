#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <locale>

using percent = float;

struct DecomposedSet {
  std::vector<std::wstring_view> intersection;
  std::vector<std::wstring_view> difference_ab;
  std::vector<std::wstring_view> difference_ba;
  DecomposedSet(std::vector<std::wstring_view> intersection, std::vector<std::wstring_view> difference_ab, std::vector<std::wstring_view> difference_ba)
    : intersection(std::move(intersection)), difference_ab(std::move(difference_ab)), difference_ba(std::move(difference_ba)) {}
};


struct Affix {
  std::size_t prefix_len;
  std::size_t suffix_len;
};

namespace utils {

  template<typename T>
  std::vector<std::wstring_view> splitSV(const T &str);

  DecomposedSet set_decomposition(std::vector<std::wstring_view> a, std::vector<std::wstring_view> b);


  std::size_t joined_size(const std::wstring_view &x);

  std::size_t joined_size(const std::vector<std::wstring_view> &x);


  std::wstring join(const std::vector<std::wstring_view> &sentence);

  percent result_cutoff(float result, percent score_cutoff);

  void trim(std::wstring &s);
  void lower_case(std::wstring &s);

  std::wstring default_process(std::wstring s);

  Affix remove_common_affix(std::wstring_view& a, std::wstring_view& b);

  void remove_common_affix(std::vector<std::wstring_view> &a, std::vector<std::wstring_view> &b);
}


template<typename T>
inline std::vector<std::wstring_view> utils::splitSV(const T &str) {
  std::vector<std::wstring_view> output;
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
