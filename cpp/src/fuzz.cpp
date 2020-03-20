#include "fuzz.hpp"
#include "levenshtein.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>


percent fuzz::partial_ratio(const std::string &a, const std::string &b, percent score_cutoff) {
  if (a.empty() || b.empty() || score_cutoff >= 100) {
    return 0;
  }

  std::string_view shorter;
  std::string_view longer;

  if (a.length() > b.length()) {
    shorter = b;
    longer = a;
  } else {
    shorter = a;
    longer = b;
  }

  auto blocks = levenshtein::matching_blocks(shorter, longer);
  float max_ratio = 0;
  for (const auto &block : blocks) {
      size_t long_start = (block.second_start > block.first_start) ? block.second_start - block.first_start : 0;
      std::string_view long_substr = longer.substr(long_start, shorter.length());

      float ls_ratio = levenshtein::normalized_weighted_distance(shorter, long_substr, score_cutoff);

      if (ls_ratio > 0.995) {
  			return 100;
  		}

      if (ls_ratio > max_ratio) {
  			max_ratio = ls_ratio;
  		}
  }

  return max_ratio * 100;
}


static percent full_ratio(const std::string &a, const std::string &b, percent score_cutoff=0) {
  const float UNBASE_SCALE = 0.95;

  float sratio = fuzz::ratio(a, b, score_cutoff);
  score_cutoff = std::max(score_cutoff, sratio)/UNBASE_SCALE;
  return std::max(sratio, fuzz::token_ratio(a, b, score_cutoff) * UNBASE_SCALE);
}


percent fuzz::ratio(const std::string &a, const std::string &b, percent score_cutoff) {
  return levenshtein::normalized_weighted_distance(a, b, score_cutoff / 100) * 100;
}


percent fuzz::token_ratio(const std::string &a, const std::string &b, percent score_cutoff) {
  if (score_cutoff >= 100) {
    return 0;
  }

  std::vector<std::string_view> tokens_a = splitSV(a);
  std::sort(tokens_a.begin(), tokens_a.end());
  std::vector<std::string_view> tokens_b = splitSV(b);
  std::sort(tokens_b.begin(), tokens_b.end());

  auto decomposition = set_decomposition(tokens_a, tokens_b);

  size_t ab_len = recursiveIterableSize(decomposition.difference_ab, 1);
  size_t ba_len = recursiveIterableSize(decomposition.difference_ba, 1);
  size_t double_prefix = 2 * recursiveIterableSize(decomposition.intersection, 1);

  // fuzzywuzzy joined sect and ab/ba for comparisions
  // this is not done here as an optimisation, so the lengths get incremented by 1
  // since there would be a whitespace between the joined strings
  if (double_prefix) {
    // exit early since this will always result in a ratio of 1
    if (!ab_len || !ba_len) return 100;

    ++ab_len;
    ++ba_len;
  }

  float result = levenshtein::normalized_weighted_distance(tokens_a, tokens_b, score_cutoff / 100, " ");
  size_t lensum = ab_len + ba_len + double_prefix;

  // TODO: could add score cutoff aswell, but would need to copy most things from normalized_score_cutoff
  // as an alternative add another utility function to levenshtein for this case
  size_t sect_distance = levenshtein::weighted_distance(decomposition.difference_ab, decomposition.difference_ba, " ");
  if (sect_distance != std::numeric_limits<size_t>::max()) {
    result = std::max(result, (float)1.0 - sect_distance / (float)lensum);
  }

  // exit early since the other ratios are 0
  // (when a or b was empty they would cause a segfault)
  if (!double_prefix) {
    return result * 100;
  }

  return std::max({
    result,
    // levenshtein distances sect+ab <-> sect and sect+ba <-> sect
    // would exit early after removing the prefix sect, so the distance can be directly calculated
    (float)1.0 - (float)ab_len / (float)(ab_len + double_prefix),
    (float)1.0 - (float)ba_len / (float)(ba_len + double_prefix)
  }) * 100;
}


// combines token_set and token_sort ratio from fuzzywuzzy so it is only required to
// do a lot of operations once
percent partial_token_ratio(const std::string &a, const std::string &b, percent score_cutoff=0.0) {
  if (score_cutoff >= 100) {
    return 0;
  }

  // TODO: probably faster to split the String view already sorted
  std::vector<std::string_view> tokens_a = splitSV(a);
  std::sort(tokens_a.begin(), tokens_a.end());
  std::vector<std::string_view> tokens_b = splitSV(b);
  std::sort(tokens_b.begin(), tokens_b.end());

  auto unique_a = tokens_a;
  auto unique_b = tokens_b;
  unique_a.erase(std::unique(unique_a.begin(), unique_a.end()), unique_a.end());
  unique_b.erase(std::unique(unique_b.begin(), unique_b.end()), unique_b.end());
    
  std::vector<std::string_view> difference_ab;
  std::vector<std::string_view> difference_ba;

  std::set_difference(unique_a.begin(), unique_a.end(), unique_b.begin(), unique_b.end(), 
                      std::inserter(difference_ab, difference_ab.begin()));
  std::set_difference(unique_b.begin(), unique_b.end(), unique_a.begin(), unique_a.end(), 
                      std::inserter(difference_ba, difference_ba.begin()));

  // exit early when there is a common word in both sequences
  if (difference_ab.size() < unique_a.size()) {
    return 100;
  }

  percent result = fuzz::partial_ratio(sentence_join(tokens_a), sentence_join(tokens_b), score_cutoff);
  // do not calculate the same partial_ratio twice
  if (tokens_a.size() == unique_a.size() && tokens_b.size() == unique_b.size()) {
    return result;
  }

  score_cutoff = std::max(score_cutoff, result);
  return std::max(
    result,
    fuzz::partial_ratio(sentence_join(difference_ab), sentence_join(difference_ba), score_cutoff)
  );
}


percent fuzz::QRatio(const std::string &a, const std::string &b, percent score_cutoff) {
  if (score_cutoff >= 100) {
    return 0;
  }

  return ratio(a, b, score_cutoff);
}


percent fuzz::WRatio(const std::string &a, const std::string &b, percent score_cutoff) {
  if (score_cutoff >= 100) {
    return 0;
  }

  const float UNBASE_SCALE = 0.95;

  size_t len_a = a.length();
  size_t len_b = b.length();
  float len_ratio = (len_a > len_b) ? (float)len_a / (float)len_b : (float)len_b / (float)len_a;

  float sratio = fuzz::ratio(a, b, score_cutoff);
  score_cutoff = std::max(score_cutoff, sratio);

  if (len_ratio < 1.5) {
    return std::max(sratio, fuzz::token_ratio(a, b, score_cutoff/UNBASE_SCALE) * UNBASE_SCALE);
  }

  float partial_scale = (len_ratio < 8.0) ? 0.9 : 0.6;

  score_cutoff /= partial_scale;
  sratio = std::max(sratio, partial_ratio(a, b, score_cutoff) * partial_scale);

  score_cutoff = std::max(score_cutoff, sratio)/UNBASE_SCALE;
  return std::max(sratio, partial_token_ratio(a, b, score_cutoff) * UNBASE_SCALE * partial_scale );
}
