#include "fuzz.hpp"
#include "levenshtein.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <tuple>
#include <iterator>

template<typename CharT>
inline percent fuzz::ratio(
    const boost::basic_string_view<CharT>& s1,
    const boost::basic_string_view<CharT>& s2,
    percent score_cutoff)
{
    double result = levenshtein::normalized_weighted_distance(s1, s2, score_cutoff / 100);
    return utils::result_cutoff(result * 100, score_cutoff);
}

template<typename CharT>
inline percent fuzz::ratio(
    const std::basic_string<CharT>& s1,
    const std::basic_string<CharT>& s2,
    percent score_cutoff)
{
    return ratio(
        boost::basic_string_view<CharT>(s1),
        boost::basic_string_view<CharT>(s2),
        score_cutoff);
}

template<typename CharT>
inline percent fuzz::partial_ratio(
    boost::basic_string_view<CharT> s1,
    boost::basic_string_view<CharT> s2,
    percent score_cutoff)
{
    if (s1.empty() || s2.empty() || score_cutoff > 100) {
        return 0;
    }

    if (s1.length() > s2.length()) {
        std::swap(s1, s2);
    }

    auto blocks = levenshtein::matching_blocks(s1, s2);
    double max_ratio = 0;
    for (const auto& block : blocks) {
        std::size_t long_start = (block.second_start > block.first_start) ? block.second_start - block.first_start : 0;
        boost::basic_string_view<CharT> long_substr = s2.substr(long_start, s1.length());

        double ls_ratio = levenshtein::normalized_weighted_distance(s1, long_substr, score_cutoff / 100);

        if (ls_ratio > 0.995) {
            return 100;
        }

        if (ls_ratio > max_ratio) {
            max_ratio = ls_ratio;
        }
    }

    return utils::result_cutoff(max_ratio * 100, score_cutoff);
}

template<typename CharT>
inline percent fuzz::partial_ratio(
    const std::basic_string<CharT>& s1,
    const std::basic_string<CharT>& s2,
    percent score_cutoff)
{
    return partial_ratio(
        boost::basic_string_view<CharT>(s1),
        boost::basic_string_view<CharT>(s2),
        score_cutoff);
}

template<typename CharT>
percent _token_sort(
    const boost::basic_string_view<CharT>& s1,
    const boost::basic_string_view<CharT>& s2,
    bool partial,
    percent score_cutoff = 0.0)
{
    if (score_cutoff > 100) {
        return 0;
    }

    string_view_vec<CharT> tokens_a = utils::splitSV(s1);
    std::sort(tokens_a.begin(), tokens_a.end());
    string_view_vec<CharT> tokens_b = utils::splitSV(s2);
    std::sort(tokens_b.begin(), tokens_b.end());

    if (partial) {
        return fuzz::partial_ratio(
            utils::join(tokens_a),
            utils::join(tokens_b),
            score_cutoff);
    }
    else {
        double result = levenshtein::normalized_weighted_distance(
            utils::join(tokens_a),
            utils::join(tokens_b),
            score_cutoff / 100);
        return utils::result_cutoff(result * 100, score_cutoff);
    }
}

template<typename CharT>
percent fuzz::token_sort_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff)
{
    return _token_sort(s1, s2, false, score_cutoff);
}

template<typename CharT>
percent fuzz::token_sort_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff)
{
    return _token_sort(
        boost::basic_string_view<CharT>(s1), 
        boost::basic_string_view<CharT>(s2), 
        false, score_cutoff);
}

template<typename CharT>
percent fuzz::partial_token_sort_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff)
{
    return _token_sort(s1, s2, true, score_cutoff);
}

template<typename CharT>
percent fuzz::partial_token_sort_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff)
{
    return _token_sort(
        boost::basic_string_view<CharT>(s1), 
        boost::basic_string_view<CharT>(s2), 
        true, score_cutoff);
}

template<typename CharT>
percent fuzz::token_set_ratio(
    const boost::basic_string_view<CharT>& s1,
    const boost::basic_string_view<CharT>& s2,
    percent score_cutoff)
{
    if (score_cutoff > 100) {
        return 0;
    }

    string_view_vec<CharT> tokens_a = utils::splitSV(s1);
    std::sort(tokens_a.begin(), tokens_a.end());
    string_view_vec<CharT> tokens_b = utils::splitSV(s2);
    std::sort(tokens_b.begin(), tokens_b.end());

    auto decomposition = utils::set_decomposition(tokens_a, tokens_b);
    auto intersection = decomposition.intersection;
    auto difference_ab = decomposition.difference_ab;
    auto difference_ba = decomposition.difference_ba;

    std::basic_string<CharT> diff_ab_joined = utils::join(difference_ab);
    std::basic_string<CharT> diff_ba_joined = utils::join(difference_ba);

    std::size_t ab_len = diff_ab_joined.length();
    std::size_t ba_len = diff_ba_joined.length();
    std::size_t sect_len = utils::joined_size(intersection);

    // exit early since this will always result in a ratio of 1
    if (sect_len && (!ab_len || !ba_len)) {
        return 100;
    }

    // string length sect+ab <-> sect and sect+ba <-> sect
    std::size_t sect_ab_lensum = sect_len + !!sect_len + ab_len;
    std::size_t sect_ba_lensum = sect_len + !!sect_len + ba_len;

    std::size_t sect_distance = levenshtein::weighted_distance(diff_ab_joined, diff_ba_joined);
    double result = 0;
    if (sect_distance != std::numeric_limits<std::size_t>::max()) {
        result = std::max(result, 1.0 - sect_distance / static_cast<double>(sect_ab_lensum + sect_ba_lensum));
    }

    // exit early since the other ratios are 0
    if (!sect_len) {
        return utils::result_cutoff(result * 100, score_cutoff);
    }

    // levenshtein distance sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    std::size_t sect_ab_distance = !!sect_len + ab_len;
    std::size_t sect_ba_distance = !!sect_len + ba_len;

    result = std::max({ result,
        1.0 - sect_ab_distance / static_cast<double>(sect_len + sect_ab_lensum),
        1.0 - sect_ba_distance / static_cast<double>(sect_len + sect_ba_lensum) });
    return utils::result_cutoff(result * 100, score_cutoff);
}

template<typename CharT>
percent fuzz::token_set_ratio(
    const std::basic_string<CharT>& s1,
    const std::basic_string<CharT>& s2,
    percent score_cutoff)
{
    return token_set_ratio(
        boost::basic_string_view<CharT>(s1), 
        boost::basic_string_view<CharT>(s2), 
        score_cutoff);
}

template<typename CharT>
percent fuzz::partial_token_set_ratio(
    const boost::basic_string_view<CharT>& s1,
    const boost::basic_string_view<CharT>& s2,
    percent score_cutoff)
{
    if (score_cutoff > 100) {
        return 0;
    }

    string_view_vec<CharT> tokens_a = utils::splitSV(s1);
    std::sort(tokens_a.begin(), tokens_a.end());
    string_view_vec<CharT> tokens_b = utils::splitSV(s2);
    std::sort(tokens_b.begin(), tokens_b.end());

    tokens_a.erase(std::unique(tokens_a.begin(), tokens_a.end()), tokens_a.end());
    tokens_b.erase(std::unique(tokens_b.begin(), tokens_b.end()), tokens_b.end());

    string_view_vec<CharT> difference_ab;
    string_view_vec<CharT> difference_ba;

    std::set_difference(tokens_a.begin(), tokens_a.end(), tokens_b.begin(), tokens_b.end(),
        std::inserter(difference_ab, difference_ab.begin()));
    std::set_difference(tokens_b.begin(), tokens_b.end(), tokens_a.begin(), tokens_a.end(),
        std::inserter(difference_ba, difference_ba.begin()));

    // exit early when there is a common word in both sequences
    if (difference_ab.size() < tokens_a.size()) {
        return 100;
    }

    return partial_ratio(utils::join(difference_ab), utils::join(difference_ba), score_cutoff);
}

template<typename CharT>
percent fuzz::partial_token_set_ratio(
    const std::basic_string<CharT>& s1,
    const std::basic_string<CharT>& s2,
    percent score_cutoff)
{
    return partial_token_set_ratio(
        boost::basic_string_view<CharT>(s1), 
        boost::basic_string_view<CharT>(s2), 
        score_cutoff);
}

template<typename CharT>
percent fuzz::token_ratio(
    const Sentence<CharT>& s1,
    const Sentence<CharT>& s2,
    percent score_cutoff)
{
    if (score_cutoff > 100) {
        return 0;
    }

    string_view_vec<CharT> tokens_a = utils::splitSV(s1.sentence);
    std::sort(tokens_a.begin(), tokens_a.end());
    string_view_vec<CharT> tokens_b = utils::splitSV(s2.sentence);
    std::sort(tokens_b.begin(), tokens_b.end());

    auto decomposition = utils::set_decomposition(tokens_a, tokens_b);
    auto intersection = decomposition.intersection;
    auto difference_ab = decomposition.difference_ab;
    auto difference_ba = decomposition.difference_ba;

    std::basic_string<CharT> diff_ab_joined = utils::join(difference_ab);
    std::basic_string<CharT> diff_ba_joined = utils::join(difference_ba);

    std::size_t ab_len = diff_ab_joined.length();
    std::size_t ba_len = diff_ba_joined.length();
    std::size_t sect_len = utils::joined_size(intersection);

    // exit early since this will always result in a ratio of 1
    if (sect_len && (!ab_len || !ba_len)) {
        return 100;
    }

    double result = 0;
    if (quick_lev_estimate(s1, s2, score_cutoff)) {
        result = levenshtein::normalized_weighted_distance(
            utils::join(tokens_a),
            utils::join(tokens_b),
            score_cutoff / 100);
    }

    // string length sect+ab <-> sect and sect+ba <-> sect
    std::size_t sect_ab_lensum = sect_len + !!sect_len + ab_len;
    std::size_t sect_ba_lensum = sect_len + !!sect_len + ba_len;

    Sentence<CharT> diff_ab{diff_ab_joined,  utils::bitmap_create(diff_ab_joined)};
    Sentence<CharT> diff_ba{diff_ba_joined,  utils::bitmap_create(diff_ba_joined)};
    double bm_ratio = 1.0 - bitmap_distance(diff_ab, diff_ba) / static_cast<double>(sect_ab_lensum + sect_ba_lensum);
    if (bm_ratio >= score_cutoff) {
        std::size_t sect_distance = levenshtein::weighted_distance(diff_ab_joined, diff_ba_joined);
        result = std::max(result, 1.0 - sect_distance / static_cast<double>(sect_ab_lensum + sect_ba_lensum));
    }

    // exit early since the other ratios are 0
    if (!sect_len) {
        return utils::result_cutoff(result * 100, score_cutoff);
    }

    // levenshtein distance sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    std::size_t sect_ab_distance = !!sect_len + ab_len;
    std::size_t sect_ba_distance = !!sect_len + ba_len;

    result = std::max({ result,
        // levenshtein distances sect+ab <-> sect and sect+ba <-> sect
        // would exit early after removing the prefix sect, so the distance can be directly calculated
        1.0 - sect_ab_distance / static_cast<double>(sect_len + sect_ab_lensum),
        1.0 - sect_ba_distance / static_cast<double>(sect_len + sect_ba_lensum) });
    return utils::result_cutoff(result * 100, score_cutoff);
}

// combines token_set and token_sort ratio from fuzzywuzzy so it is only required to
// do a lot of operations once
template<typename CharT>
percent fuzz::partial_token_ratio(
    const boost::basic_string_view<CharT>& s1,
    const boost::basic_string_view<CharT>& s2,
    percent score_cutoff)
{
    if (score_cutoff > 100) {
        return 0;
    }

    std::vector<boost::wstring_view> tokens_a = utils::splitSV(s1);
    std::sort(tokens_a.begin(), tokens_a.end());
    std::vector<boost::wstring_view> tokens_b = utils::splitSV(s2);
    std::sort(tokens_b.begin(), tokens_b.end());

    auto unique_a = tokens_a;
    auto unique_b = tokens_b;
    unique_a.erase(std::unique(unique_a.begin(), unique_a.end()), unique_a.end());
    unique_b.erase(std::unique(unique_b.begin(), unique_b.end()), unique_b.end());

    std::vector<boost::wstring_view> difference_ab;
    std::vector<boost::wstring_view> difference_ba;

    std::set_difference(unique_a.begin(), unique_a.end(), unique_b.begin(), unique_b.end(),
        std::inserter(difference_ab, difference_ab.begin()));
    std::set_difference(unique_b.begin(), unique_b.end(), unique_a.begin(), unique_a.end(),
        std::inserter(difference_ba, difference_ba.begin()));

    // exit early when there is a common word in both sequences
    if (difference_ab.size() < unique_a.size()) {
        return 100;
    }

    percent result = partial_ratio(utils::join(tokens_a), utils::join(tokens_b), score_cutoff);
    // do not calculate the same partial_ratio twice
    if (tokens_a.size() == unique_a.size() && tokens_b.size() == unique_b.size()) {
        return result;
    }

    score_cutoff = std::max(score_cutoff, result);
    return std::max(
        result,
        partial_ratio(utils::join(difference_ab), utils::join(difference_ba), score_cutoff));
}

template<typename CharT>
percent fuzz::partial_token_ratio(
    const std::basic_string<CharT>& s1,
    const std::basic_string<CharT>& s2,
    percent score_cutoff)
{
    return partial_token_ratio(
        boost::basic_string_view<CharT>(s1),
        boost::basic_string_view<CharT>(s2),
        score_cutoff);
}

template<typename CharT>
std::size_t fuzz::bitmap_distance(const Sentence<CharT>& s1, const Sentence<CharT>& s2)
{
    uint64_t bitmap1 = s1.bitmap;
    uint64_t bitmap2 = s2.bitmap;

    std::size_t distance = 0;
    while (bitmap1 || bitmap2) {
        uint8_t val1 = bitmap1 & 0b1111;
        uint8_t val2 = bitmap2 & 0b1111;
        distance += std::abs(val1 - val2);
        bitmap1 >>= 4;
        bitmap2 >>= 4;
    }
    return distance;
}

template<typename CharT>
percent fuzz::bitmap_ratio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff)
{
    std::size_t distance = bitmap_distance(s1, s2);
    std::size_t lensum = s1.sentence.length() + s2.sentence.length();
    percent result = 1.0 - static_cast<double>(distance) / lensum;

    return utils::result_cutoff(result * 100, score_cutoff);
}


template<typename CharT>
percent fuzz::length_ratio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff)
{
    std::size_t s1_len = s1.sentence.length();
    std::size_t s2_len = s2.sentence.length();
    std::size_t distance = (s1_len > s2_len)
        ? s1_len - s2_len
        : s2_len - s1_len;
    
    std::size_t lensum = s1_len + s2_len;
    double result = 1.0 - static_cast<double>(distance) / lensum;
    return utils::result_cutoff(result * 100, score_cutoff);
}

template<typename CharT>
percent fuzz::quick_lev_estimate(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff)
{
    if (s1.bitmap || s2.bitmap) {
        return bitmap_ratio(s1, s2, score_cutoff);
    } else {
        return length_ratio(s1, s2, score_cutoff);
    }
}

template<typename CharT>
percent fuzz::WRatio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff)
{
    if (score_cutoff > 100) {
        return 0;
    }

    const double UNBASE_SCALE = 0.95;

    std::size_t len_a = s1.sentence.length();
    std::size_t len_b = s2.sentence.length();
    double len_ratio = (len_a > len_b) ? static_cast<double>(len_a) / len_b : static_cast<double>(len_b) / len_a;

    double sratio = 0;
    if (quick_lev_estimate(s1, s2, score_cutoff)) {
        sratio = ratio(s1.sentence, s2.sentence, score_cutoff);
        // increase the score_cutoff by a small step so it might be able to exit early
        score_cutoff = std::max(score_cutoff, sratio + 0.00001);
    }

    if (len_ratio < 1.5) {
        return std::max(sratio, token_ratio(s1, s2, score_cutoff / UNBASE_SCALE) * UNBASE_SCALE);
    }

    double partial_scale = (len_ratio < 8.0) ? 0.9 : 0.6;

    score_cutoff /= partial_scale;
    sratio = std::max(sratio, partial_ratio(s1.sentence, s2.sentence, score_cutoff) * partial_scale);

    // increase the score_cutoff by a small step so it might be able to exit early
    score_cutoff = std::max(score_cutoff, sratio + 0.00001) / UNBASE_SCALE;
    return std::max(sratio, partial_token_ratio(s1.sentence, s2.sentence, score_cutoff) * UNBASE_SCALE * partial_scale);
}
