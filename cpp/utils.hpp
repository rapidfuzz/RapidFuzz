#pragma once
#include <string_view>
#include <vector>
#include <algorithm>

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
OutputIterator3 set_decomposition(InputIterator1 first1, InputIterator1 last1,
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

std::vector<std::string_view> splitSV(std::string_view str,
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

// attention this is changing b !!!!
Intersection intersection_count_sorted_vec(std::vector<std::string_view> a,
                                           std::vector<std::string_view> b) {
  std::vector<std::string_view> vec_sect;
  std::vector<std::string_view> vec_ab;

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

void remove_common_affix(std::string_view &a, std::string_view &b) {
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
}

void remove_common_affix(std::vector<std::string_view> &a,
                         std::vector<std::string_view> &b) {
  // remove common prefix
  // maybe erasing whole prefix at once is faster
  auto a_it = a.begin();
  auto b_it = b.begin();
  while (a_it != a.end() && b_it != b.end()) {
    size_t common_len = 0;
    auto a_letter_it = a_it->begin();
    auto b_letter_it = b_it->begin();
    while (a_letter_it != a_it->end() && b_letter_it != b_it->end() &&
           *a_letter_it == *b_letter_it) {
      ++a_letter_it;
      ++b_letter_it;
      ++common_len;
    }
    if (a_letter_it != a_it->end() || b_letter_it != b_it->end()) {
      *a_it = a_it->substr(common_len);
      *b_it = b_it->substr(common_len);
      break;
    }
    a_it = a.erase(a_it);
    b_it = b.erase(b_it);
  }

  // remove common suffix
  auto a_it_rev = a.rbegin();
  auto b_it_rev = b.rbegin();
  while (a_it_rev != a.rend() && b_it_rev != b.rend()) {
    size_t common_len = 0;
    auto a_letter_it = a_it_rev->rbegin();
    auto b_letter_it = b_it_rev->rbegin();
    while (a_letter_it != a_it_rev->rend() && b_letter_it != b_it_rev->rend() &&
           *a_letter_it == *b_letter_it) {
      ++a_letter_it;
      ++b_letter_it;
      ++common_len;
    }
    if (a_letter_it != a_it_rev->rend() || b_letter_it != b_it_rev->rend()) {
      *a_it_rev = a_it_rev->substr(0, a_it_rev->size() - common_len);
      *b_it_rev = b_it_rev->substr(0, b_it_rev->size() - common_len);
      break;
    }
    ++a_it_rev;
    ++b_it_rev;
    a.pop_back();
    b.pop_back();
  }
}

size_t joinedStringViewLength(const std::vector<std::string_view> &words) {
  if (words.empty()) {
    return 0;
  }
  // init length with whitespaces between words
  size_t length = words.size() - 1;
  for (const auto &word : words) {
    length += word.length();
  }
  return length;
}