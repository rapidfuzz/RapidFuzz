#include "utils.hpp"
#include <algorithm>
#include <locale>
#include <regex>

template<typename CharT>
string_view_vec<CharT> utils::splitSV(const boost::basic_string_view<CharT>& str)
{
    string_view_vec<CharT> output;

    auto first = str.data(), second = str.data(), last = first + str.size();
    for (; second != last && first != last; first = second + 1) {
        second = std::find_if(first, last, [](const CharT& c) {
            return std::isspace(c, std::locale(""));
            });

        if (first != second) {
            output.emplace_back(first, second - first);
        }
    }

    return output;
}

template<typename CharT>
string_view_vec<CharT> splitSV(const std::basic_string<CharT>& str)
{
    return splitSV(boost::basic_string_view<CharT>(str));
}

template<typename CharT>
std::size_t utils::joined_size(const string_view_vec<CharT>& x)
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

template<typename CharT>
std::basic_string<CharT> utils::join(const string_view_vec<CharT>& sentence)
{
    if (sentence.empty()) {
        return std::basic_string<CharT>();
    }

    auto sentence_iter = sentence.begin();
    std::basic_string<CharT> result{ *sentence_iter };
    const std::basic_string<CharT> whitespace{ 0x20 };
    ++sentence_iter;
    for (; sentence_iter != sentence.end(); ++sentence_iter) {
        result.append(whitespace).append(std::basic_string<CharT>{ *sentence_iter });
    }
    return result;
}

template<typename CharT>
DecomposedSet<CharT> utils::set_decomposition(string_view_vec<CharT> a, string_view_vec<CharT> b)
{
    string_view_vec<CharT> intersection;
    string_view_vec<CharT> difference_ab;
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

    return DecomposedSet<CharT>{ intersection, difference_ab, b };
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
std::size_t remove_common_prefix(boost::basic_string_view<CharT>& a, boost::basic_string_view<CharT>& b)
{
    auto prefix = common_prefix_length(a.begin(), a.end(), b.begin(), b.end());
    a.remove_prefix(prefix);
    b.remove_prefix(prefix);
    return prefix;
}

/**
 * Removes common suffix of two string views
 */
template<typename CharT>
std::size_t remove_common_suffix(boost::basic_string_view<CharT>& a, boost::basic_string_view<CharT>& b)
{
    auto suffix = common_prefix_length(a.rbegin(), a.rend(), b.rbegin(), b.rend());
    a.remove_suffix(suffix);
    b.remove_suffix(suffix);
    return suffix;
}

/**
 * Removes common affix of two string views
 */
template<typename CharT>
Affix utils::remove_common_affix(boost::basic_string_view<CharT>& a, boost::basic_string_view<CharT>& b)
{
    return Affix{
        remove_common_prefix(a, b),
        remove_common_suffix(a, b)
    };
}

template<typename CharT>
void ltrim(std::basic_string<CharT>& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](const CharT& ch) {
            return !std::isspace(ch, std::locale(""));
        }));
}

template<typename CharT>
void rtrim(std::basic_string<CharT>& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const CharT& ch) {
            return !std::isspace(ch, std::locale(""));
        }).base(), s.end());
}

template<typename CharT>
void utils::trim(std::basic_string<CharT>& s)
{
    ltrim(s);
    rtrim(s);
}

template<typename CharT>
void utils::lower_case(std::basic_string<CharT>& s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

template<typename CharT>
std::basic_string<CharT> utils::default_process(std::basic_string<CharT> s)
{
    std::basic_regex<CharT> alnum_re(CHARTYPE_STR(CharT, "[^[:alnum:]]"));
    s = std::regex_replace(s, alnum_re, CHARTYPE_STR(CharT, " "));
    trim(s);
    lower_case(s);
    return s;
}

template<typename CharT>
uint64_t utils::bitmap_create(const boost::basic_string_view<CharT>& sentence)
{
    uint64_t bitmap = 0;
    for (const unsigned int& letter : sentence) {
        uint8_t shift = (letter % 16) * 4;

        // make sure there is no overflow when more than 8 characters
        // with the same shift exist
        uint64_t bitmask = static_cast<uint64_t>(0b1111) << shift;
        if ((bitmap & bitmask) != bitmask) {
            bitmap += static_cast<uint64_t>(1) << shift;
        }
    }
    return bitmap;
}

template<typename CharT>
uint64_t utils::bitmap_create(const std::basic_string<CharT>& sentence)
{
    return bitmap_create(boost::basic_string_view<CharT>(sentence));
}

inline percent utils::result_cutoff(double result, percent score_cutoff)
{
    return (result >= score_cutoff) ? result : 0;
}