#pragma once
#include "levenshtein.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>


float token_ratio(const std::string &a, const std::string &b, float score_cutoff=0.0) {
  std::vector<std::string_view> tokens_a = splitSV(a);
  std::sort(tokens_a.begin(), tokens_a.end());
  std::vector<std::string_view> tokens_b = splitSV(b);
  std::sort(tokens_b.begin(), tokens_b.end());

  float result = levenshtein::normalized_weighted_distance(tokens_a, tokens_b, score_cutoff, " ");

  tokens_a.erase(std::unique(tokens_a.begin(), tokens_a.end()), tokens_a.end());
  tokens_b.erase(std::unique(tokens_b.begin(), tokens_b.end()), tokens_b.end());

  auto intersection = intersection_count_sorted_vec(tokens_a, tokens_b);

  size_t ab_len = recursiveIterableSize(intersection.ab, 1);
  size_t ba_len = recursiveIterableSize(intersection.ba, 1);
  size_t double_prefix = 2 * recursiveIterableSize(intersection.sect, 1);

  // fuzzywuzzy joined sect and ab/ba for comparisions
  // this is not done here as an optimisation, so the lengths get incremented by 1
  // since there would be a whitespace between the joined strings
  if (double_prefix) {
    // exit early since this will always result in a ratio of 1
    if (!ab_len || !ba_len) return 1.0;

    ++ab_len;
    ++ba_len;
  }

  size_t lensum = ab_len + ba_len + double_prefix;
  size_t sect_distance = levenshtein::weighted_distance(intersection.ab, intersection.ba, score_cutoff, " ");
  float sect_result = (sect_distance >= lensum)
    ? (float)0.0
    : (float)1.0 - sect_distance / (float)lensum

  // exit early since the other ratios are 0
  if (!double_prefix) {
    return std::max(result, sect_result);
  }

  return std::max({
    result,
    sect_result,
    // levenshtein distances sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    (float)1.0 - (float)ab_len / (float)(ab_len + double_prefix),
    (float)1.0 - (float)ba_len / (float)(ba_len + double_prefix)
  });
}


float full_ratio(const std::string &query, const std::string &choice,
                   float score_cutoff)
{
  float normalized_score_cutoff = score_cutoff / (float)100.0;

  // this needs more thoughts when to start using score cutoff, since it performs slower when it can not exit early
  // has to be tested with some real training examples
  float sratio = (normalized_score_cutoff > 0.7)
    ? levenshtein::normalized_weighted_distance(query, choice, normalized_score_cutoff)
    : levenshtein::normalized_weighted_distance(query, choice);

  const float UNBASE_SCALE = 0.95;
  float min_ratio = std::max(normalized_score_cutoff, sratio);
  if (min_ratio < UNBASE_SCALE) {
    sratio = (normalized_score_cutoff > 0.7)
      ? std::max(sratio, token_ratio(query, choice, normalized_score_cutoff/UNBASE_SCALE) * UNBASE_SCALE)
      : std::max(sratio, token_ratio(query, choice) * UNBASE_SCALE);
  }
  return sratio * 100.0;
}


/*uint8_t partial_ratio(const std::string &query, const std::string &choice,
                      uint8_t partial_scale, uint8_t score_cutoff)
{
  float sratio = normalized_levenshtein(query, choice);
  float min_ratio = std::max(sratio, (float)score_cutoff / (float)100);
  if (min_ratio < partial_scale) {
    sratio = std::max(sratio, partial_string_ratio(query, choice) * partial_scale);
    min_ratio = std::max(sratio, min_ratio);
    const float UNBASE_SCALE = 0.95;
    if (min_ratio < UNBASE_SCALE * partial_scale) {
      sratio = std::max(sratio, partial_token_ratio(query, choice) * UNBASE_SCALE * partial_scale );
    }
  }
  return static_cast<uint8_t>(std::round(sratio * 100.0));
}*/


uint8_t ratio(const std::string &query, const std::string &choice,
              uint8_t score_cutoff)
{
  if (query == choice) {
    return 100;
  }

  if (query.empty() || choice.empty() || score_cutoff == 100) {
    return 0;
  }

  size_t len_a = query.length();
  size_t len_b = choice.length();
  float len_ratio = (len_a > len_b) ? (float)len_a / (float)len_b : (float)len_b / (float)len_a;

  if (len_ratio < 1.5) {
    return full_ratio(query, choice, score_cutoff);
  // TODO: this is still missing
  } else if (len_ratio < 8.0) {
    return 0.0;
    // return partial_ratio(query, choice, 0.9, score_cutoff);
  } else {
    return 0.0;
    // return partial_ratio(query, choice, 0.6, score_cutoff);
  }
}
