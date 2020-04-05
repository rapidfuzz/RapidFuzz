#pragma once
#include <boost/utility/string_view.hpp>
#include "utils.hpp"

namespace fuzz {
template<typename CharT>
percent ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_ratio(
	boost::basic_string_view<CharT> s1,
	boost::basic_string_view<CharT> s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent token_sort_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent token_sort_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_sort_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_sort_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent token_set_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent token_set_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_set_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_set_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent token_ratio(
	const Sentence<CharT>& s1,
	const Sentence<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_ratio(
	const boost::basic_string_view<CharT>& s1,
	const boost::basic_string_view<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
percent partial_token_ratio(
	const std::basic_string<CharT>& s1,
	const std::basic_string<CharT>& s2,
	percent score_cutoff = 0);

template<typename CharT>
std::size_t bitmap_distance(const Sentence<CharT>& s1, const Sentence<CharT>& s2);

template<typename CharT>
percent bitmap_ratio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff = 0);

template<typename CharT>
percent length_ratio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff = 0);

template<typename CharT>
percent quick_lev_estimate(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff = 0);

template<typename CharT>
percent WRatio(const Sentence<CharT>& s1, const Sentence<CharT>& s2, percent score_cutoff = 0);
}

#include "fuzz.txx"
