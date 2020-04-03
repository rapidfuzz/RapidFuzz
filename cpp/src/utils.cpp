#include "utils.hpp"
#include <algorithm>
#include <cwctype>

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
std::size_t remove_common_prefix(std::wstring_view& a, std::wstring_view& b)
{
    auto prefix = common_prefix_length(a.begin(), a.end(), b.begin(), b.end());
    a.remove_prefix(prefix);
    b.remove_prefix(prefix);
    return prefix;
}

/**
 * Removes common suffix of two string views
 */
std::size_t remove_common_suffix(std::wstring_view& a, std::wstring_view& b)
{
    auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
    a.remove_suffix(suffix);
    b.remove_suffix(suffix);
    return suffix;
}

/**
 * Removes common affix of two string views
 */
Affix utils::remove_common_affix(std::wstring_view& a, std::wstring_view& b)
{
    return Affix{
        remove_common_prefix(a, b),
        remove_common_suffix(a, b)
    };
}

template <typename T>
void vec_remove_common_affix(T& a, T& b)
{
    auto prefix = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
    a.erase(a.begin(), prefix.first);
    b.erase(b.begin(), prefix.second);

    auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
    a.erase(a.end() - suffix, a.end());
    b.erase(b.end() - suffix, b.end());
}

void utils::remove_common_affix(std::vector<std::wstring_view>& a, std::vector<std::wstring_view>& b)
{
    vec_remove_common_affix(a, b);
    if (!a.empty() && !b.empty()) {
        remove_common_prefix(a.front(), b.front());
        remove_common_suffix(a.back(), b.back());
    }
}

std::wstring utils::join(const std::vector<std::wstring_view>& sentence)
{
    if (sentence.empty()) {
        return std::wstring();
    }

    auto sentence_iter = sentence.begin();
    std::wstring result{ *sentence_iter };
    const std::wstring whitespace{ 0x20 };
    ++sentence_iter;
    for (; sentence_iter != sentence.end(); ++sentence_iter) {
        result.append(whitespace).append(std::wstring{ *sentence_iter });
    }
    return result;
}

percent utils::result_cutoff(double result, percent score_cutoff)
{
    return (result >= score_cutoff) ? result : 0;
}

// trim from start (in place)
void ltrim(std::wstring& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](const wchar_t &ch) {
                return !std::iswspace(ch);
            }));
}

// trim from end (in place)
void rtrim(std::wstring& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const wchar_t &ch) {
                return !std::iswspace(ch);
            }).base(), s.end());
}

// trim from both ends (in place)
void utils::trim(std::wstring& s)
{
    ltrim(s);
    rtrim(s);
}

void utils::lower_case(std::wstring& s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

std::wstring utils::default_process(std::wstring s)
{
    // replace embedded null terminators
    std::replace( s.begin(), s.end(), {'\x00'}, ' ');
    trim(s);
    lower_case(s);
    return s;
}

DecomposedSet utils::set_decomposition(std::vector<std::wstring_view> a, std::vector<std::wstring_view> b)
{
    std::vector<std::wstring_view> intersection;
    std::vector<std::wstring_view> difference_ab;
    a.erase(std::unique(a.begin(), a.end()), a.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());

    for (const auto& current_a : a) {
        auto element_b = std::find(b.begin(), b.end(), current_a);
        if (element_b != b.end()) {
            b.erase(element_b);
            intersection.emplace_back(current_a);
        } else {
            difference_ab.emplace_back(current_a);
        }
    }

    return DecomposedSet{ intersection, difference_ab, b };
}

std::size_t utils::joined_size(const std::wstring_view& x)
{
    return x.size();
}

std::size_t utils::joined_size(const std::vector<std::wstring_view>& x)
{
    if (x.empty()) {
        return 0;
    }

    // there is a whitespace between each word
    std::size_t result = x.size() - 1;
    for (const auto& y : x) {
        result += y.size();
    }

    return result;
}

std::vector<std::wstring_view> utils::splitSV(const std::wstring_view& str)
{
    std::vector<std::wstring_view> output;
    // assume a word length of 6 + 1 whitespace
    output.reserve(str.size() / 7);

    auto first = str.data(), second = str.data(), last = first + str.size();
    for (; second != last && first != last; first = second + 1) {
        // maybe use localisation
        second = std::find_if(first, last, [](const wchar_t &c) { return std::iswspace(c); });

        if (first != second) {
            output.emplace_back(first, second - first);
        }
    }

    return output;
}
